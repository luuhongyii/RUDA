use super::*;

impl BoolTensorOps<Self> for Dispatch {
    fn bool_autodiff_context(tensor: &BoolTensor<Self>) -> ruda_tensor::AutodiffContext {
        tensor.context()
    }

    fn bool_with_autodiff_context(tensor: BoolTensor<Self>, context: ruda_tensor::AutodiffContext) -> BoolTensor<Self> {
        tensor.with_context(context)
    }

    fn bool_empty(shape: Shape, device: &Device<Self>, dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::factory(device);
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_empty(shape, $device, dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_zeros(shape: Shape, device: &Device<Self>, dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::factory(device);
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_zeros(shape, $device, dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_ones(shape: Shape, device: &Device<Self>, dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::factory(device);
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_ones(shape, $device, dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_into_data( tensor: BoolTensor<Self>, ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let future = <$B>::bool_into_data(unpack!($mode, $variant, Bool, tensor, $context));
                let future: core::pin::Pin<Box<dyn Future<Output = Result<TensorData, ExecutionError>> + Send>> = Box::pin(async move { future.await });
                future
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_from_data(data: TensorData, device: &Device<Self>) -> BoolTensor<Self> {
        let selection = Selection::factory(device);
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_from_data(data, $device);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_into_int(tensor: BoolTensor<Self>, out_dtype: IntDType) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_into_int(unpack!($mode, $variant, Bool, tensor, $context), out_dtype);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_into_float(tensor: BoolTensor<Self>, out_dtype: FloatDType) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_into_float(unpack!($mode, $variant, Bool, tensor, $context), out_dtype);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_device(tensor: &BoolTensor<Self>) -> Device<Self> {
        tensor.device()
    }

    fn bool_to_device(tensor: BoolTensor<Self>, device: &Device<Self>) -> BoolTensor<Self> {
        crate::transfer::bool_to_device(tensor, device)
    }

    fn bool_reshape(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_reshape(unpack!($mode, $variant, Bool, tensor, $context), shape);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_slice(tensor: BoolTensor<Self>, slices: &[Slice]) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_slice(unpack!($mode, $variant, Bool, tensor, $context), slices);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_slice_assign( tensor: BoolTensor<Self>, slices: &[Slice], value: BoolTensor<Self>, ) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&value)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_slice_assign(unpack!($mode, $variant, Bool, tensor, $context), slices, unpack!($mode, $variant, Bool, value, $context));
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_mask_where( tensor: BoolTensor<Self>, mask: BoolTensor<Self>, value: BoolTensor<Self>, ) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&mask)).chain(core::iter::once(&value)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_mask_where(unpack!($mode, $variant, Bool, tensor, $context), unpack!($mode, $variant, Bool, mask, $context), unpack!($mode, $variant, Bool, value, $context));
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_mask_fill(tensor: BoolTensor<Self>, mask: BoolTensor<Self>, value: Scalar) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&mask)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_mask_fill(unpack!($mode, $variant, Bool, tensor, $context), unpack!($mode, $variant, Bool, mask, $context), value);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_gather(dim: usize, tensor: BoolTensor<Self>, indices: IntTensor<Self>) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&indices)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_gather(dim, unpack!($mode, $variant, Bool, tensor, $context), unpack!($mode, $variant, Int, indices, $context));
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_scatter_or( dim: usize, tensor: BoolTensor<Self>, indices: IntTensor<Self>, value: BoolTensor<Self>, ) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&indices)).chain(core::iter::once(&value)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_scatter_or(dim, unpack!($mode, $variant, Bool, tensor, $context), unpack!($mode, $variant, Int, indices, $context), unpack!($mode, $variant, Bool, value, $context));
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_select(tensor: BoolTensor<Self>, dim: usize, indices: IntTensor<Self>) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&indices)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_select(unpack!($mode, $variant, Bool, tensor, $context), dim, unpack!($mode, $variant, Int, indices, $context));
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_select_or( tensor: BoolTensor<Self>, dim: usize, indices: IntTensor<Self>, value: BoolTensor<Self>, ) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&indices)).chain(core::iter::once(&value)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_select_or(unpack!($mode, $variant, Bool, tensor, $context), dim, unpack!($mode, $variant, Int, indices, $context), unpack!($mode, $variant, Bool, value, $context));
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_repeat_dim(tensor: BoolTensor<Self>, dim: usize, times: usize) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_repeat_dim(unpack!($mode, $variant, Bool, tensor, $context), dim, times);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_cat(tensors: Vec<BoolTensor<Self>>, dim: usize) -> BoolTensor<Self> {
        let selection = Selection::tensors(tensors.iter());
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_cat(tensors.into_iter().map(|value| unpack!($mode, $variant, Bool, value, $context)).collect(), dim);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_equal(unpack!($mode, $variant, Bool, lhs, $context), unpack!($mode, $variant, Bool, rhs, $context));
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_not_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_not_equal(unpack!($mode, $variant, Bool, lhs, $context), unpack!($mode, $variant, Bool, rhs, $context));
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_equal_elem(lhs: BoolTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_equal_elem(unpack!($mode, $variant, Bool, lhs, $context), rhs);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_not_equal_elem(lhs: BoolTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_not_equal_elem(unpack!($mode, $variant, Bool, lhs, $context), rhs);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_not(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_not(unpack!($mode, $variant, Bool, tensor, $context));
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_and(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_and(unpack!($mode, $variant, Bool, lhs, $context), unpack!($mode, $variant, Bool, rhs, $context));
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_or(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_or(unpack!($mode, $variant, Bool, lhs, $context), unpack!($mode, $variant, Bool, rhs, $context));
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_xor(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_xor(unpack!($mode, $variant, Bool, lhs, $context), unpack!($mode, $variant, Bool, rhs, $context));
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_transpose(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_transpose(unpack!($mode, $variant, Bool, tensor, $context));
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_swap_dims(tensor: BoolTensor<Self>, dim1: usize, dim2: usize) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_swap_dims(unpack!($mode, $variant, Bool, tensor, $context), dim1, dim2);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_permute(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_permute(unpack!($mode, $variant, Bool, tensor, $context), axes);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_flip(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_flip(unpack!($mode, $variant, Bool, tensor, $context), axes);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_any(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_any(unpack!($mode, $variant, Bool, tensor, $context));
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_any_dim(tensor: BoolTensor<Self>, dim: usize) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_any_dim(unpack!($mode, $variant, Bool, tensor, $context), dim);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_all(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_all(unpack!($mode, $variant, Bool, tensor, $context));
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_all_dim(tensor: BoolTensor<Self>, dim: usize) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_all_dim(unpack!($mode, $variant, Bool, tensor, $context), dim);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_argwhere( tensor: BoolTensor<Self>, out_dtype: IntDType, ) -> impl Future<Output = IntTensor<Self>> + 'static + Send {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let future = <$B>::bool_argwhere(unpack!($mode, $variant, Bool, tensor, $context), out_dtype);
                let future: core::pin::Pin<Box<dyn Future<Output = IntTensor<Self>> + Send>> = Box::pin(async move { pack!($mode, $variant, Int, future.await) });
                future
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_expand(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_expand(unpack!($mode, $variant, Bool, tensor, $context), shape);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bool_unfold(tensor: BoolTensor<Self>, dim: usize, size: usize, step: usize) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bool_unfold(unpack!($mode, $variant, Bool, tensor, $context), dim, size, step);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }
}
