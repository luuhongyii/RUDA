use super::*;

impl QTensorOps<Self> for Dispatch {
    fn q_from_data(data: TensorData, device: &Device<Self>) -> QuantizedTensor<Self> {
        let selection = Selection::factory(device);
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_from_data(data, $device);
                pack!($mode, $variant, Quantized, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn quantize( tensor: FloatTensor<Self>, scheme: &QuantScheme, qparams: QuantizationParametersPrimitive<Self>, ) -> QuantizedTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&qparams.scales)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::quantize(unpack!($mode, $variant, Float, tensor, $context), scheme, QuantizationParametersPrimitive { scales: unpack!($mode, $variant, Float, qparams.scales, $context) });
                pack!($mode, $variant, Quantized, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn quantize_dynamic(tensor: FloatTensor<Self>, scheme: &QuantScheme) -> QuantizedTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::quantize_dynamic(unpack!($mode, $variant, Float, tensor, $context), scheme);
                pack!($mode, $variant, Quantized, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn quantize_dynamic_with_precision(tensor: FloatTensor<Self>, scheme: &QuantScheme, calibration_dtype: FloatDType) -> QuantizedTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::quantize_dynamic_with_precision(unpack!($mode, $variant, Float, tensor, $context), scheme, calibration_dtype);
                pack!($mode, $variant, Quantized, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn dequantize(tensor: QuantizedTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::dequantize(unpack!($mode, $variant, Quantized, tensor, $context), dtype);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_device(tensor: &QuantizedTensor<Self>) -> Device<Self> {
        tensor.device()
    }

    fn q_to_device(tensor: QuantizedTensor<Self>, device: &Device<Self>) -> QuantizedTensor<Self> {
        crate::transfer::q_to_device(tensor, device)
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_reshape(unpack!($mode, $variant, Quantized, tensor, $context), shape);
                pack!($mode, $variant, Quantized, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_into_data( tensor: QuantizedTensor<Self>, ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let future = <$B>::q_into_data(unpack!($mode, $variant, Quantized, tensor, $context));
                let future: core::pin::Pin<Box<dyn Future<Output = Result<TensorData, ExecutionError>> + Send>> = Box::pin(async move { future.await });
                future
            }};
        }
        dispatch!(selection, call)
    }

    fn q_detach(tensor: QuantizedTensor<Self>) -> QuantizedTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_detach(unpack!($mode, $variant, Quantized, tensor, $context));
                pack!($mode, $variant, Quantized, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_set_require_grad(tensor: QuantizedTensor<Self>, _require_grad: bool) -> QuantizedTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_set_require_grad(unpack!($mode, $variant, Quantized, tensor, $context), _require_grad);
                pack!($mode, $variant, Quantized, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_is_require_grad(_tensor: &QuantizedTensor<Self>) -> bool {
        let selection = Selection::tensors(core::iter::once(_tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_is_require_grad(borrow_tensor!($mode, $variant, Quantized, _tensor));
                result
            }};
        }
        dispatch!(selection, call)
    }

    fn q_is_tracked(tensor: &QuantizedTensor<Self>) -> bool {
        let selection = Selection::tensors(core::iter::once(tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_is_tracked(borrow_tensor!($mode, $variant, Quantized, tensor));
                result
            }};
        }
        dispatch!(selection, call)
    }

    fn q_gradient_checkpointing_strategy(_tensor: &QuantizedTensor<Self>) -> Option<ruda_tensor::GradientCheckpointingStrategy> {
        match _tensor.context() { AutodiffContext::Disabled => None, AutodiffContext::Enabled(strategy) => Some(strategy) }
    }

    fn q_autodiff_context(tensor: &QuantizedTensor<Self>) -> ruda_tensor::AutodiffContext {
        tensor.context()
    }

    fn q_with_autodiff_context(tensor: QuantizedTensor<Self>, context: ruda_tensor::AutodiffContext) -> QuantizedTensor<Self> {
        tensor.with_context(context)
    }

    fn q_with_gradient_checkpointing_strategy( _tensor: QuantizedTensor<Self>, _strategy: ruda_tensor::GradientCheckpointingStrategy, ) -> QuantizedTensor<Self> {
        assert!(matches!(_tensor.context(), AutodiffContext::Enabled(_)), "Checkpointing requires autodiff");
        _tensor.with_context(AutodiffContext::Enabled(_strategy))
    }

    fn q_expand(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_expand(unpack!($mode, $variant, Quantized, tensor, $context), shape);
                pack!($mode, $variant, Quantized, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_transpose(tensor: QuantizedTensor<Self>) -> QuantizedTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_transpose(unpack!($mode, $variant, Quantized, tensor, $context));
                pack!($mode, $variant, Quantized, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_swap_dims(tensor: QuantizedTensor<Self>, dim1: usize, dim2: usize) -> QuantizedTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_swap_dims(unpack!($mode, $variant, Quantized, tensor, $context), dim1, dim2);
                pack!($mode, $variant, Quantized, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_permute(unpack!($mode, $variant, Quantized, tensor, $context), axes);
                pack!($mode, $variant, Quantized, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_flip(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_flip(unpack!($mode, $variant, Quantized, tensor, $context), axes);
                pack!($mode, $variant, Quantized, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_select( tensor: QuantizedTensor<Self>, dim: usize, indices: IntTensor<Self>, ) -> QuantizedTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&indices)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_select(unpack!($mode, $variant, Quantized, tensor, $context), dim, unpack!($mode, $variant, Int, indices, $context));
                pack!($mode, $variant, Quantized, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_slice(tensor: QuantizedTensor<Self>, slices: &[Slice]) -> QuantizedTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_slice(unpack!($mode, $variant, Quantized, tensor, $context), slices);
                pack!($mode, $variant, Quantized, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_gather( dim: usize, tensor: QuantizedTensor<Self>, indices: IntTensor<Self>, ) -> QuantizedTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&indices)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_gather(dim, unpack!($mode, $variant, Quantized, tensor, $context), unpack!($mode, $variant, Int, indices, $context));
                pack!($mode, $variant, Quantized, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_repeat_dim(tensor: QuantizedTensor<Self>, dim: usize, times: usize) -> QuantizedTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_repeat_dim(unpack!($mode, $variant, Quantized, tensor, $context), dim, times);
                pack!($mode, $variant, Quantized, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_add(lhs: QuantizedTensor<Self>, rhs: QuantizedTensor<Self>) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_add(unpack!($mode, $variant, Quantized, lhs, $context), unpack!($mode, $variant, Quantized, rhs, $context));
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_add_scalar(lhs: QuantizedTensor<Self>, rhs: Scalar) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_add_scalar(unpack!($mode, $variant, Quantized, lhs, $context), rhs);
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_clamp_min(tensor: QuantizedTensor<Self>, min: Scalar) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_clamp_min(unpack!($mode, $variant, Quantized, tensor, $context), min);
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_clamp_max(tensor: QuantizedTensor<Self>, max: Scalar) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_clamp_max(unpack!($mode, $variant, Quantized, tensor, $context), max);
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_clamp(tensor: QuantizedTensor<Self>, min: Scalar, max: Scalar) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_clamp(unpack!($mode, $variant, Quantized, tensor, $context), min, max);
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_sub(lhs: QuantizedTensor<Self>, rhs: QuantizedTensor<Self>) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_sub(unpack!($mode, $variant, Quantized, lhs, $context), unpack!($mode, $variant, Quantized, rhs, $context));
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_sub_scalar(lhs: QuantizedTensor<Self>, rhs: Scalar) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_sub_scalar(unpack!($mode, $variant, Quantized, lhs, $context), rhs);
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_mul(lhs: QuantizedTensor<Self>, rhs: QuantizedTensor<Self>) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_mul(unpack!($mode, $variant, Quantized, lhs, $context), unpack!($mode, $variant, Quantized, rhs, $context));
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_mul_scalar(lhs: QuantizedTensor<Self>, rhs: Scalar) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_mul_scalar(unpack!($mode, $variant, Quantized, lhs, $context), rhs);
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_div(lhs: QuantizedTensor<Self>, rhs: QuantizedTensor<Self>) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_div(unpack!($mode, $variant, Quantized, lhs, $context), unpack!($mode, $variant, Quantized, rhs, $context));
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_div_scalar(lhs: QuantizedTensor<Self>, rhs: Scalar) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_div_scalar(unpack!($mode, $variant, Quantized, lhs, $context), rhs);
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_matmul(lhs: TensorPrimitive<Self>, rhs: TensorPrimitive<Self>) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(match &lhs { TensorPrimitive::Float(tensor) | TensorPrimitive::QFloat(tensor) => tensor }).chain(core::iter::once(match &rhs { TensorPrimitive::Float(tensor) | TensorPrimitive::QFloat(tensor) => tensor })));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_matmul(match lhs { TensorPrimitive::Float(value) => TensorPrimitive::Float(unpack!($mode, $variant, Float, value, $context)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(unpack!($mode, $variant, Quantized, value, $context)) }, match rhs { TensorPrimitive::Float(value) => TensorPrimitive::Float(unpack!($mode, $variant, Float, value, $context)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(unpack!($mode, $variant, Quantized, value, $context)) });
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_matmul_default(lhs: TensorPrimitive<Self>, rhs: TensorPrimitive<Self>) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(match &lhs { TensorPrimitive::Float(tensor) | TensorPrimitive::QFloat(tensor) => tensor }).chain(core::iter::once(match &rhs { TensorPrimitive::Float(tensor) | TensorPrimitive::QFloat(tensor) => tensor })));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_matmul_default(match lhs { TensorPrimitive::Float(value) => TensorPrimitive::Float(unpack!($mode, $variant, Float, value, $context)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(unpack!($mode, $variant, Quantized, value, $context)) }, match rhs { TensorPrimitive::Float(value) => TensorPrimitive::Float(unpack!($mode, $variant, Float, value, $context)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(unpack!($mode, $variant, Quantized, value, $context)) });
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_neg(tensor: QuantizedTensor<Self>) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_neg(unpack!($mode, $variant, Quantized, tensor, $context));
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_recip(tensor: QuantizedTensor<Self>) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_recip(unpack!($mode, $variant, Quantized, tensor, $context));
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_sum(tensor: QuantizedTensor<Self>) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_sum(unpack!($mode, $variant, Quantized, tensor, $context));
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_sum_dim(tensor: QuantizedTensor<Self>, dim: usize) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_sum_dim(unpack!($mode, $variant, Quantized, tensor, $context), dim);
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_prod(tensor: QuantizedTensor<Self>) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_prod(unpack!($mode, $variant, Quantized, tensor, $context));
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_prod_dim(tensor: QuantizedTensor<Self>, dim: usize) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_prod_dim(unpack!($mode, $variant, Quantized, tensor, $context), dim);
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_mean(tensor: QuantizedTensor<Self>) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_mean(unpack!($mode, $variant, Quantized, tensor, $context));
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_mean_dim(tensor: QuantizedTensor<Self>, dim: usize) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_mean_dim(unpack!($mode, $variant, Quantized, tensor, $context), dim);
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_cumsum(tensor: QuantizedTensor<Self>, dim: usize) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_cumsum(unpack!($mode, $variant, Quantized, tensor, $context), dim);
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_cumprod(tensor: QuantizedTensor<Self>, dim: usize) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_cumprod(unpack!($mode, $variant, Quantized, tensor, $context), dim);
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_cummin(tensor: QuantizedTensor<Self>, dim: usize) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_cummin(unpack!($mode, $variant, Quantized, tensor, $context), dim);
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_cummax(tensor: QuantizedTensor<Self>, dim: usize) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_cummax(unpack!($mode, $variant, Quantized, tensor, $context), dim);
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_exp(tensor: QuantizedTensor<Self>) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_exp(unpack!($mode, $variant, Quantized, tensor, $context));
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_log(tensor: QuantizedTensor<Self>) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_log(unpack!($mode, $variant, Quantized, tensor, $context));
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_log1p(tensor: QuantizedTensor<Self>) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_log1p(unpack!($mode, $variant, Quantized, tensor, $context));
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_powf(lhs: QuantizedTensor<Self>, rhs: QuantizedTensor<Self>) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_powf(unpack!($mode, $variant, Quantized, lhs, $context), unpack!($mode, $variant, Quantized, rhs, $context));
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_powi(lhs: QuantizedTensor<Self>, rhs: IntTensor<Self>) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_powi(unpack!($mode, $variant, Quantized, lhs, $context), unpack!($mode, $variant, Int, rhs, $context));
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_powi_scalar(lhs: QuantizedTensor<Self>, rhs: Scalar) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_powi_scalar(unpack!($mode, $variant, Quantized, lhs, $context), rhs);
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_powf_scalar(tensor: QuantizedTensor<Self>, value: Scalar) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_powf_scalar(unpack!($mode, $variant, Quantized, tensor, $context), value);
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_sqrt(tensor: QuantizedTensor<Self>) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_sqrt(unpack!($mode, $variant, Quantized, tensor, $context));
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_abs(tensor: QuantizedTensor<Self>) -> QuantizedTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_abs(unpack!($mode, $variant, Quantized, tensor, $context));
                pack!($mode, $variant, Quantized, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_cos(tensor: QuantizedTensor<Self>) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_cos(unpack!($mode, $variant, Quantized, tensor, $context));
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_sin(tensor: QuantizedTensor<Self>) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_sin(unpack!($mode, $variant, Quantized, tensor, $context));
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_tan(tensor: QuantizedTensor<Self>) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_tan(unpack!($mode, $variant, Quantized, tensor, $context));
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_cosh(tensor: QuantizedTensor<Self>) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_cosh(unpack!($mode, $variant, Quantized, tensor, $context));
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_sinh(tensor: QuantizedTensor<Self>) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_sinh(unpack!($mode, $variant, Quantized, tensor, $context));
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_tanh(tensor: QuantizedTensor<Self>) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_tanh(unpack!($mode, $variant, Quantized, tensor, $context));
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_erf(tensor: QuantizedTensor<Self>) -> TensorPrimitive<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_erf(unpack!($mode, $variant, Quantized, tensor, $context));
                match result { TensorPrimitive::Float(value) => TensorPrimitive::Float(pack!($mode, $variant, Float, value)), TensorPrimitive::QFloat(value) => TensorPrimitive::QFloat(pack!($mode, $variant, Quantized, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_cat(tensors: Vec<QuantizedTensor<Self>>, dim: usize) -> QuantizedTensor<Self> {
        let selection = Selection::tensors(tensors.iter());
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_cat(tensors.into_iter().map(|value| unpack!($mode, $variant, Quantized, value, $context)).collect(), dim);
                pack!($mode, $variant, Quantized, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_argmax(tensor: QuantizedTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_argmax(unpack!($mode, $variant, Quantized, tensor, $context), dim, out_dtype);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_argtopk( tensor: QuantizedTensor<Self>, dim: usize, k: usize, out_dtype: IntDType, ) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_argtopk(unpack!($mode, $variant, Quantized, tensor, $context), dim, k, out_dtype);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_topk(tensor: QuantizedTensor<Self>, dim: usize, k: usize) -> QuantizedTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_topk(unpack!($mode, $variant, Quantized, tensor, $context), dim, k);
                pack!($mode, $variant, Quantized, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_argmin(tensor: QuantizedTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_argmin(unpack!($mode, $variant, Quantized, tensor, $context), dim, out_dtype);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_max(tensor: QuantizedTensor<Self>) -> QuantizedTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_max(unpack!($mode, $variant, Quantized, tensor, $context));
                pack!($mode, $variant, Quantized, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_max_dim(tensor: QuantizedTensor<Self>, dim: usize) -> QuantizedTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_max_dim(unpack!($mode, $variant, Quantized, tensor, $context), dim);
                pack!($mode, $variant, Quantized, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_max_dim_with_indices( tensor: QuantizedTensor<Self>, dim: usize, out_dtype: IntDType, ) -> (QuantizedTensor<Self>, IntTensor<Self>) {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_max_dim_with_indices(unpack!($mode, $variant, Quantized, tensor, $context), dim, out_dtype);
                { let (value0, value1) = result; (pack!($mode, $variant, Quantized, value0), pack!($mode, $variant, Int, value1)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_min(tensor: QuantizedTensor<Self>) -> QuantizedTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_min(unpack!($mode, $variant, Quantized, tensor, $context));
                pack!($mode, $variant, Quantized, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_min_dim(tensor: QuantizedTensor<Self>, dim: usize) -> QuantizedTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_min_dim(unpack!($mode, $variant, Quantized, tensor, $context), dim);
                pack!($mode, $variant, Quantized, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_min_dim_with_indices( tensor: QuantizedTensor<Self>, dim: usize, out_dtype: IntDType, ) -> (QuantizedTensor<Self>, IntTensor<Self>) {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_min_dim_with_indices(unpack!($mode, $variant, Quantized, tensor, $context), dim, out_dtype);
                { let (value0, value1) = result; (pack!($mode, $variant, Quantized, value0), pack!($mode, $variant, Int, value1)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_max_abs(tensor: QuantizedTensor<Self>) -> QuantizedTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_max_abs(unpack!($mode, $variant, Quantized, tensor, $context));
                pack!($mode, $variant, Quantized, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_max_abs_dim(tensor: QuantizedTensor<Self>, dim: usize) -> QuantizedTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_max_abs_dim(unpack!($mode, $variant, Quantized, tensor, $context), dim);
                pack!($mode, $variant, Quantized, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_any(tensor: QuantizedTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_any(unpack!($mode, $variant, Quantized, tensor, $context), out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_any_dim(tensor: QuantizedTensor<Self>, dim: usize, out_dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_any_dim(unpack!($mode, $variant, Quantized, tensor, $context), dim, out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_all(tensor: QuantizedTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_all(unpack!($mode, $variant, Quantized, tensor, $context), out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_all_dim(tensor: QuantizedTensor<Self>, dim: usize, out_dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_all_dim(unpack!($mode, $variant, Quantized, tensor, $context), dim, out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_sort(tensor: QuantizedTensor<Self>, dim: usize, descending: bool) -> QuantizedTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_sort(unpack!($mode, $variant, Quantized, tensor, $context), dim, descending);
                pack!($mode, $variant, Quantized, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn q_sort_with_indices( tensor: QuantizedTensor<Self>, dim: usize, descending: bool, out_dtype: IntDType, ) -> (QuantizedTensor<Self>, IntTensor<Self>) {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_sort_with_indices(unpack!($mode, $variant, Quantized, tensor, $context), dim, descending, out_dtype);
                { let (value0, value1) = result; (pack!($mode, $variant, Quantized, value0), pack!($mode, $variant, Int, value1)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn q_argsort( tensor: QuantizedTensor<Self>, dim: usize, descending: bool, out_dtype: IntDType, ) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::q_argsort(unpack!($mode, $variant, Quantized, tensor, $context), dim, descending, out_dtype);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }
}
