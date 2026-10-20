use super::*;

impl ActivationOps<Self> for Dispatch {
    fn silu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::silu(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn leaky_relu(tensor: FloatTensor<Self>, negative_slope: Scalar) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::leaky_relu(unpack!($mode, $variant, Float, tensor, $context), negative_slope);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn relu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::relu(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn relu_backward(output: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&output).chain(core::iter::once(&grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::relu_backward(unpack!($mode, $variant, Float, output, $context), unpack!($mode, $variant, Float, grad, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn gelu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::gelu(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn prelu(tensor: FloatTensor<Self>, alpha: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&alpha)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::prelu(unpack!($mode, $variant, Float, tensor, $context), unpack!($mode, $variant, Float, alpha, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn gelu_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::gelu_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, grad, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn sigmoid(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::sigmoid(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn sigmoid_backward(output: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&output).chain(core::iter::once(&grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::sigmoid_backward(unpack!($mode, $variant, Float, output, $context), unpack!($mode, $variant, Float, grad, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn hard_sigmoid(tensor: FloatTensor<Self>, alpha: Scalar, beta: Scalar) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::hard_sigmoid(unpack!($mode, $variant, Float, tensor, $context), alpha, beta);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn log_sigmoid(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::log_sigmoid(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn softmax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::softmax(unpack!($mode, $variant, Float, tensor, $context), dim);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn log_softmax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::log_softmax(unpack!($mode, $variant, Float, tensor, $context), dim);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn softmin(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::softmin(unpack!($mode, $variant, Float, tensor, $context), dim);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn log_sigmoid_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::log_sigmoid_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, grad, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }
}
