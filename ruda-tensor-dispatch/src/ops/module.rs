use super::*;

impl ModuleOps<Self> for Dispatch {
    fn batch_norm( x: FloatTensor<Self>, gamma: FloatTensor<Self>, beta: FloatTensor<Self>, mean: FloatTensor<Self>, variance: FloatTensor<Self>, epsilon: f64, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&gamma)).chain(core::iter::once(&beta)).chain(core::iter::once(&mean)).chain(core::iter::once(&variance)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::batch_norm(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, gamma, $context), unpack!($mode, $variant, Float, beta, $context), unpack!($mode, $variant, Float, mean, $context), unpack!($mode, $variant, Float, variance, $context), epsilon);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn batch_norm_train( x: FloatTensor<Self>, gamma: FloatTensor<Self>, beta: FloatTensor<Self>, epsilon: f64, ) -> BatchNormTrain<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&gamma)).chain(core::iter::once(&beta)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::batch_norm_train(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, gamma, $context), unpack!($mode, $variant, Float, beta, $context), epsilon);
                BatchNormTrain { output: pack!($mode, $variant, Float, result.output), mean: pack!($mode, $variant, Float, result.mean), variance: pack!($mode, $variant, Float, result.variance) }
            }};
        }
        dispatch!(selection, call)
    }

    fn batch_norm_train_backward( x: FloatTensor<Self>, gamma: FloatTensor<Self>, mean: FloatTensor<Self>, variance: FloatTensor<Self>, epsilon: f64, output_grad: FloatTensor<Self>, ) -> BatchNormTrainBackward<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&gamma)).chain(core::iter::once(&mean)).chain(core::iter::once(&variance)).chain(core::iter::once(&output_grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::batch_norm_train_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, gamma, $context), unpack!($mode, $variant, Float, mean, $context), unpack!($mode, $variant, Float, variance, $context), epsilon, unpack!($mode, $variant, Float, output_grad, $context));
                BatchNormTrainBackward { x_grad: pack!($mode, $variant, Float, result.x_grad), gamma_grad: pack!($mode, $variant, Float, result.gamma_grad), beta_grad: pack!($mode, $variant, Float, result.beta_grad) }
            }};
        }
        dispatch!(selection, call)
    }

    fn exponential_relu_native(tensor: FloatTensor<Self>, alpha: f64, continuous: bool) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::exponential_relu_native(unpack!($mode, $variant, Float, tensor, $context), alpha, continuous);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn exponential_relu_native_backward(tensor: FloatTensor<Self>, grad: FloatTensor<Self>, alpha: f64, continuous: bool) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::exponential_relu_native_backward(unpack!($mode, $variant, Float, tensor, $context), unpack!($mode, $variant, Float, grad, $context), alpha, continuous);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn leaky_relu_native(tensor: FloatTensor<Self>, negative_slope: f64) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::leaky_relu_native(unpack!($mode, $variant, Float, tensor, $context), negative_slope);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn leaky_relu_native_backward(tensor: FloatTensor<Self>, grad: FloatTensor<Self>, negative_slope: f64) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::leaky_relu_native_backward(unpack!($mode, $variant, Float, tensor, $context), unpack!($mode, $variant, Float, grad, $context), negative_slope);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn prelu_native(tensor: FloatTensor<Self>, alpha: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&alpha)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::prelu_native(unpack!($mode, $variant, Float, tensor, $context), unpack!($mode, $variant, Float, alpha, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn prelu_native_backward_select(tensor: FloatTensor<Self>, alpha: FloatTensor<Self>, grad: FloatTensor<Self>, mask: [bool; 2]) -> [Option<FloatTensor<Self>>; 2] {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&alpha)).chain(core::iter::once(&grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::prelu_native_backward_select(unpack!($mode, $variant, Float, tensor, $context), unpack!($mode, $variant, Float, alpha, $context), unpack!($mode, $variant, Float, grad, $context), mask);
                result.map(|value| value.map(|value| pack!($mode, $variant, Float, value)))
            }};
        }
        dispatch!(selection, call)
    }

    fn group_norm(tensor: FloatTensor<Self>, gamma: Option<FloatTensor<Self>>, beta: Option<FloatTensor<Self>>, groups: usize, epsilon: f64) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(gamma.iter()).chain(beta.iter()));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::group_norm(unpack!($mode, $variant, Float, tensor, $context), gamma.map(|value| unpack!($mode, $variant, Float, value, $context)), beta.map(|value| unpack!($mode, $variant, Float, value, $context)), groups, epsilon);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn group_norm_with_stats(tensor: FloatTensor<Self>, gamma: Option<FloatTensor<Self>>, beta: Option<FloatTensor<Self>>, groups: usize, epsilon: f64) -> LayerNormOutput<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(gamma.iter()).chain(beta.iter()));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::group_norm_with_stats(unpack!($mode, $variant, Float, tensor, $context), gamma.map(|value| unpack!($mode, $variant, Float, value, $context)), beta.map(|value| unpack!($mode, $variant, Float, value, $context)), groups, epsilon);
                LayerNormOutput { output: pack!($mode, $variant, Float, result.output), mean: pack!($mode, $variant, Float, result.mean), rstd: pack!($mode, $variant, Float, result.rstd) }
            }};
        }
        dispatch!(selection, call)
    }

    fn group_norm_backward_select(tensor: FloatTensor<Self>, gamma: Option<FloatTensor<Self>>, grad: FloatTensor<Self>, mean: FloatTensor<Self>, rstd: FloatTensor<Self>, groups: usize, mask: [bool; 3]) -> [Option<FloatTensor<Self>>; 3] {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(gamma.iter()).chain(core::iter::once(&grad)).chain(core::iter::once(&mean)).chain(core::iter::once(&rstd)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::group_norm_backward_select(unpack!($mode, $variant, Float, tensor, $context), gamma.map(|value| unpack!($mode, $variant, Float, value, $context)), unpack!($mode, $variant, Float, grad, $context), unpack!($mode, $variant, Float, mean, $context), unpack!($mode, $variant, Float, rstd, $context), groups, mask);
                result.map(|value| value.map(|value| pack!($mode, $variant, Float, value)))
            }};
        }
        dispatch!(selection, call)
    }

    fn gelu_native(tensor: FloatTensor<Self>, approximate: bool) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::gelu_native(unpack!($mode, $variant, Float, tensor, $context), approximate);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn gelu_native_backward(input: FloatTensor<Self>, grad: FloatTensor<Self>, approximate: bool) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&input).chain(core::iter::once(&grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::gelu_native_backward(unpack!($mode, $variant, Float, input, $context), unpack!($mode, $variant, Float, grad, $context), approximate);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn silu_native(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::silu_native(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn silu_native_backward(input: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&input).chain(core::iter::once(&grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::silu_native_backward(unpack!($mode, $variant, Float, input, $context), unpack!($mode, $variant, Float, grad, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn softmax_native(tensor: FloatTensor<Self>, dim: usize, logarithmic: bool) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::softmax_native(unpack!($mode, $variant, Float, tensor, $context), dim, logarithmic);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn softmax_with_stats(tensor: FloatTensor<Self>, dim: usize, logarithmic: bool) -> SoftmaxOutput<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::softmax_with_stats(unpack!($mode, $variant, Float, tensor, $context), dim, logarithmic);
                SoftmaxOutput { output: pack!($mode, $variant, Float, result.output), working: pack!($mode, $variant, Float, result.working) }
            }};
        }
        dispatch!(selection, call)
    }

    fn softmax_native_backward(working: FloatTensor<Self>, grad: FloatTensor<Self>, dim: usize, logarithmic: bool) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&working).chain(core::iter::once(&grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::softmax_native_backward(unpack!($mode, $variant, Float, working, $context), unpack!($mode, $variant, Float, grad, $context), dim, logarithmic);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn embedding(weights: FloatTensor<Self>, indices: IntTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&weights).chain(core::iter::once(&indices)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::embedding(unpack!($mode, $variant, Float, weights, $context), unpack!($mode, $variant, Int, indices, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn embedding_backward( weights: FloatTensor<Self>, output_grad: FloatTensor<Self>, indices: IntTensor<Self>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&weights).chain(core::iter::once(&output_grad)).chain(core::iter::once(&indices)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::embedding_backward(unpack!($mode, $variant, Float, weights, $context), unpack!($mode, $variant, Float, output_grad, $context), unpack!($mode, $variant, Int, indices, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn linear( x: FloatTensor<Self>, weight: FloatTensor<Self>, bias: Option<FloatTensor<Self>>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&weight)).chain(bias.iter()));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::linear(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, weight, $context), bias.map(|value| unpack!($mode, $variant, Float, value, $context)));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn linear_x_backward(weight: FloatTensor<Self>, output_grad: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&weight).chain(core::iter::once(&output_grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::linear_x_backward(unpack!($mode, $variant, Float, weight, $context), unpack!($mode, $variant, Float, output_grad, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn linear_weight_backward(x: FloatTensor<Self>, output_grad: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&output_grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::linear_weight_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, output_grad, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn linear_bias_backward(output_grad: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&output_grad));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::linear_bias_backward(unpack!($mode, $variant, Float, output_grad, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn conv1d( x: FloatTensor<Self>, weight: FloatTensor<Self>, bias: Option<FloatTensor<Self>>, options: ConvOptions<1>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&weight)).chain(bias.iter()));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::conv1d(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, weight, $context), bias.map(|value| unpack!($mode, $variant, Float, value, $context)), options);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn conv1d_x_backward( x: FloatTensor<Self>, weight: FloatTensor<Self>, output_grad: FloatTensor<Self>, options: ConvOptions<1>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&weight)).chain(core::iter::once(&output_grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::conv1d_x_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, weight, $context), unpack!($mode, $variant, Float, output_grad, $context), options);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn conv1d_weight_backward( x: FloatTensor<Self>, weight: FloatTensor<Self>, output_grad: FloatTensor<Self>, options: ConvOptions<1>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&weight)).chain(core::iter::once(&output_grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::conv1d_weight_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, weight, $context), unpack!($mode, $variant, Float, output_grad, $context), options);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn conv1d_bias_backward( x: FloatTensor<Self>, bias: FloatTensor<Self>, output_grad: FloatTensor<Self>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&bias)).chain(core::iter::once(&output_grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::conv1d_bias_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, bias, $context), unpack!($mode, $variant, Float, output_grad, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn conv2d( x: FloatTensor<Self>, weight: FloatTensor<Self>, bias: Option<FloatTensor<Self>>, options: ConvOptions<2>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&weight)).chain(bias.iter()));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::conv2d(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, weight, $context), bias.map(|value| unpack!($mode, $variant, Float, value, $context)), options);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn conv2d_x_backward( x: FloatTensor<Self>, weight: FloatTensor<Self>, output_grad: FloatTensor<Self>, options: ConvOptions<2>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&weight)).chain(core::iter::once(&output_grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::conv2d_x_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, weight, $context), unpack!($mode, $variant, Float, output_grad, $context), options);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn conv2d_weight_backward( x: FloatTensor<Self>, weight: FloatTensor<Self>, output_grad: FloatTensor<Self>, options: ConvOptions<2>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&weight)).chain(core::iter::once(&output_grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::conv2d_weight_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, weight, $context), unpack!($mode, $variant, Float, output_grad, $context), options);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn conv2d_bias_backward( x: FloatTensor<Self>, bias: FloatTensor<Self>, output_grad: FloatTensor<Self>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&bias)).chain(core::iter::once(&output_grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::conv2d_bias_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, bias, $context), unpack!($mode, $variant, Float, output_grad, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn deform_conv2d( x: FloatTensor<Self>, offset: FloatTensor<Self>, weight: FloatTensor<Self>, mask: Option<FloatTensor<Self>>, bias: Option<FloatTensor<Self>>, options: DeformConvOptions<2>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&offset)).chain(core::iter::once(&weight)).chain(mask.iter()).chain(bias.iter()));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::deform_conv2d(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, offset, $context), unpack!($mode, $variant, Float, weight, $context), mask.map(|value| unpack!($mode, $variant, Float, value, $context)), bias.map(|value| unpack!($mode, $variant, Float, value, $context)), options);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn deform_conv2d_backward( x: FloatTensor<Self>, offset: FloatTensor<Self>, weight: FloatTensor<Self>, mask: Option<FloatTensor<Self>>, bias: Option<FloatTensor<Self>>, output_grad: FloatTensor<Self>, options: DeformConvOptions<2>, ) -> DeformConv2dBackward<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&offset)).chain(core::iter::once(&weight)).chain(mask.iter()).chain(bias.iter()).chain(core::iter::once(&output_grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::deform_conv2d_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, offset, $context), unpack!($mode, $variant, Float, weight, $context), mask.map(|value| unpack!($mode, $variant, Float, value, $context)), bias.map(|value| unpack!($mode, $variant, Float, value, $context)), unpack!($mode, $variant, Float, output_grad, $context), options);
                DeformConv2dBackward { x_grad: pack!($mode, $variant, Float, result.x_grad), offset_grad: pack!($mode, $variant, Float, result.offset_grad), weight_grad: pack!($mode, $variant, Float, result.weight_grad), mask_grad: result.mask_grad.map(|value| pack!($mode, $variant, Float, value)), bias_grad: result.bias_grad.map(|value| pack!($mode, $variant, Float, value)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn conv3d( x: FloatTensor<Self>, weight: FloatTensor<Self>, bias: Option<FloatTensor<Self>>, options: ConvOptions<3>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&weight)).chain(bias.iter()));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::conv3d(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, weight, $context), bias.map(|value| unpack!($mode, $variant, Float, value, $context)), options);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn conv3d_x_backward( x: FloatTensor<Self>, weight: FloatTensor<Self>, output_grad: FloatTensor<Self>, options: ConvOptions<3>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&weight)).chain(core::iter::once(&output_grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::conv3d_x_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, weight, $context), unpack!($mode, $variant, Float, output_grad, $context), options);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn conv3d_weight_backward( x: FloatTensor<Self>, weight: FloatTensor<Self>, output_grad: FloatTensor<Self>, options: ConvOptions<3>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&weight)).chain(core::iter::once(&output_grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::conv3d_weight_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, weight, $context), unpack!($mode, $variant, Float, output_grad, $context), options);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn conv3d_bias_backward( x: FloatTensor<Self>, bias: FloatTensor<Self>, output_grad: FloatTensor<Self>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&bias)).chain(core::iter::once(&output_grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::conv3d_bias_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, bias, $context), unpack!($mode, $variant, Float, output_grad, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn conv_transpose1d( x: FloatTensor<Self>, weight: FloatTensor<Self>, bias: Option<FloatTensor<Self>>, options: ConvTransposeOptions<1>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&weight)).chain(bias.iter()));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::conv_transpose1d(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, weight, $context), bias.map(|value| unpack!($mode, $variant, Float, value, $context)), options);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn conv_transpose1d_x_backward( weight: FloatTensor<Self>, output_grad: FloatTensor<Self>, options: ConvTransposeOptions<1>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&weight).chain(core::iter::once(&output_grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::conv_transpose1d_x_backward(unpack!($mode, $variant, Float, weight, $context), unpack!($mode, $variant, Float, output_grad, $context), options);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn conv_transpose1d_weight_backward( x: FloatTensor<Self>, weight: FloatTensor<Self>, output_grad: FloatTensor<Self>, options: ConvTransposeOptions<1>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&weight)).chain(core::iter::once(&output_grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::conv_transpose1d_weight_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, weight, $context), unpack!($mode, $variant, Float, output_grad, $context), options);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn conv_transpose1d_bias_backward( x: FloatTensor<Self>, bias: FloatTensor<Self>, output_grad: FloatTensor<Self>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&bias)).chain(core::iter::once(&output_grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::conv_transpose1d_bias_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, bias, $context), unpack!($mode, $variant, Float, output_grad, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn conv_transpose2d( x: FloatTensor<Self>, weight: FloatTensor<Self>, bias: Option<FloatTensor<Self>>, options: ConvTransposeOptions<2>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&weight)).chain(bias.iter()));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::conv_transpose2d(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, weight, $context), bias.map(|value| unpack!($mode, $variant, Float, value, $context)), options);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn conv_transpose2d_x_backward( weight: FloatTensor<Self>, output_grad: FloatTensor<Self>, options: ConvTransposeOptions<2>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&weight).chain(core::iter::once(&output_grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::conv_transpose2d_x_backward(unpack!($mode, $variant, Float, weight, $context), unpack!($mode, $variant, Float, output_grad, $context), options);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn conv_transpose2d_weight_backward( x: FloatTensor<Self>, weight: FloatTensor<Self>, output_grad: FloatTensor<Self>, options: ConvTransposeOptions<2>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&weight)).chain(core::iter::once(&output_grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::conv_transpose2d_weight_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, weight, $context), unpack!($mode, $variant, Float, output_grad, $context), options);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn conv_transpose2d_bias_backward( x: FloatTensor<Self>, bias: FloatTensor<Self>, output_grad: FloatTensor<Self>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&bias)).chain(core::iter::once(&output_grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::conv_transpose2d_bias_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, bias, $context), unpack!($mode, $variant, Float, output_grad, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn conv_transpose3d( x: FloatTensor<Self>, weight: FloatTensor<Self>, bias: Option<FloatTensor<Self>>, options: ConvTransposeOptions<3>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&weight)).chain(bias.iter()));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::conv_transpose3d(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, weight, $context), bias.map(|value| unpack!($mode, $variant, Float, value, $context)), options);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn conv_transpose3d_x_backward( weight: FloatTensor<Self>, output_grad: FloatTensor<Self>, options: ConvTransposeOptions<3>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&weight).chain(core::iter::once(&output_grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::conv_transpose3d_x_backward(unpack!($mode, $variant, Float, weight, $context), unpack!($mode, $variant, Float, output_grad, $context), options);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn conv_transpose3d_weight_backward( x: FloatTensor<Self>, weight: FloatTensor<Self>, output_grad: FloatTensor<Self>, options: ConvTransposeOptions<3>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&weight)).chain(core::iter::once(&output_grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::conv_transpose3d_weight_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, weight, $context), unpack!($mode, $variant, Float, output_grad, $context), options);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn conv_transpose3d_bias_backward( x: FloatTensor<Self>, bias: FloatTensor<Self>, output_grad: FloatTensor<Self>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&bias)).chain(core::iter::once(&output_grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::conv_transpose3d_bias_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, bias, $context), unpack!($mode, $variant, Float, output_grad, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn unfold4d( x: FloatTensor<Self>, kernel_size: [usize; 2], options: UnfoldOptions, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::unfold4d(unpack!($mode, $variant, Float, x, $context), kernel_size, options);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn fold4d( x: FloatTensor<Self>, output_size: [usize; 2], kernel_size: [usize; 2], options: UnfoldOptions, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::fold4d(unpack!($mode, $variant, Float, x, $context), output_size, kernel_size, options);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn avg_pool1d( x: FloatTensor<Self>, kernel_size: usize, stride: usize, padding: usize, count_include_pad: bool, ceil_mode: bool, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::avg_pool1d(unpack!($mode, $variant, Float, x, $context), kernel_size, stride, padding, count_include_pad, ceil_mode);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn avg_pool1d_backward( x: FloatTensor<Self>, grad: FloatTensor<Self>, kernel_size: usize, stride: usize, padding: usize, count_include_pad: bool, ceil_mode: bool, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::avg_pool1d_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, grad, $context), kernel_size, stride, padding, count_include_pad, ceil_mode);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn avg_pool2d( x: FloatTensor<Self>, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2], count_include_pad: bool, ceil_mode: bool, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::avg_pool2d(unpack!($mode, $variant, Float, x, $context), kernel_size, stride, padding, count_include_pad, ceil_mode);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn avg_pool2d_backward( x: FloatTensor<Self>, grad: FloatTensor<Self>, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2], count_include_pad: bool, ceil_mode: bool, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::avg_pool2d_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, grad, $context), kernel_size, stride, padding, count_include_pad, ceil_mode);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn avg_pool3d(x: FloatTensor<Self>, kernel_size: [usize; 3], stride: [usize; 3], padding: [usize; 3], count_include_pad: bool, ceil_mode: bool) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::avg_pool3d(unpack!($mode, $variant, Float, x, $context), kernel_size, stride, padding, count_include_pad, ceil_mode);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn avg_pool3d_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>, kernel_size: [usize; 3], stride: [usize; 3], padding: [usize; 3], count_include_pad: bool, ceil_mode: bool) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::avg_pool3d_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, grad, $context), kernel_size, stride, padding, count_include_pad, ceil_mode);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn adaptive_avg_pool2d(x: FloatTensor<Self>, output_size: [usize; 2]) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::adaptive_avg_pool2d(unpack!($mode, $variant, Float, x, $context), output_size);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn adaptive_avg_pool2d_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::adaptive_avg_pool2d_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, grad, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn adaptive_avg_pool3d(x: FloatTensor<Self>, output_size: [usize; 3]) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::adaptive_avg_pool3d(unpack!($mode, $variant, Float, x, $context), output_size);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn adaptive_avg_pool3d_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::adaptive_avg_pool3d_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, grad, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn max_pool3d(x: FloatTensor<Self>, kernel: [usize; 3], stride: [usize; 3], padding: [usize; 3], dilation: [usize; 3], ceil: bool) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::max_pool3d(unpack!($mode, $variant, Float, x, $context), kernel, stride, padding, dilation, ceil);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn max_pool3d_with_indices(x: FloatTensor<Self>, kernel: [usize; 3], stride: [usize; 3], padding: [usize; 3], dilation: [usize; 3], ceil: bool) -> MaxPool3dWithIndices<Self> {
        let selection = Selection::tensors(core::iter::once(&x));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::max_pool3d_with_indices(unpack!($mode, $variant, Float, x, $context), kernel, stride, padding, dilation, ceil);
                MaxPool3dWithIndices { output: pack!($mode, $variant, Float, result.output), indices: pack!($mode, $variant, Int, result.indices) }
            }};
        }
        dispatch!(selection, call)
    }

    fn max_pool3d_with_indices_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>, indices: IntTensor<Self>, _kernel: [usize; 3], _stride: [usize; 3], _padding: [usize; 3], _dilation: [usize; 3], _ceil: bool) -> MaxPool3dBackward<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&grad)).chain(core::iter::once(&indices)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::max_pool3d_with_indices_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, grad, $context), unpack!($mode, $variant, Int, indices, $context), _kernel, _stride, _padding, _dilation, _ceil);
                MaxPool3dBackward { x_grad: pack!($mode, $variant, Float, result.x_grad) }
            }};
        }
        dispatch!(selection, call)
    }

    fn adaptive_avg_pool1d(x: FloatTensor<Self>, output_size: usize) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::adaptive_avg_pool1d(unpack!($mode, $variant, Float, x, $context), output_size);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn adaptive_avg_pool1d_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::adaptive_avg_pool1d_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, grad, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn max_pool1d( x: FloatTensor<Self>, kernel_size: usize, stride: usize, padding: usize, dilation: usize, ceil_mode: bool, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::max_pool1d(unpack!($mode, $variant, Float, x, $context), kernel_size, stride, padding, dilation, ceil_mode);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn max_pool1d_with_indices( x: FloatTensor<Self>, kernel_size: usize, stride: usize, padding: usize, dilation: usize, ceil_mode: bool, ) -> MaxPool1dWithIndices<Self> {
        let selection = Selection::tensors(core::iter::once(&x));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::max_pool1d_with_indices(unpack!($mode, $variant, Float, x, $context), kernel_size, stride, padding, dilation, ceil_mode);
                MaxPool1dWithIndices { output: pack!($mode, $variant, Float, result.output), indices: pack!($mode, $variant, Int, result.indices) }
            }};
        }
        dispatch!(selection, call)
    }

    fn max_pool1d_with_indices_backward( x: FloatTensor<Self>, kernel_size: usize, stride: usize, padding: usize, dilation: usize, ceil_mode: bool, output_grad: FloatTensor<Self>, indices: IntTensor<Self>, ) -> MaxPool1dBackward<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&output_grad)).chain(core::iter::once(&indices)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::max_pool1d_with_indices_backward(unpack!($mode, $variant, Float, x, $context), kernel_size, stride, padding, dilation, ceil_mode, unpack!($mode, $variant, Float, output_grad, $context), unpack!($mode, $variant, Int, indices, $context));
                MaxPool1dBackward { x_grad: pack!($mode, $variant, Float, result.x_grad) }
            }};
        }
        dispatch!(selection, call)
    }

    fn max_pool2d( x: FloatTensor<Self>, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2], dilation: [usize; 2], ceil_mode: bool, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::max_pool2d(unpack!($mode, $variant, Float, x, $context), kernel_size, stride, padding, dilation, ceil_mode);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn max_pool2d_with_indices( x: FloatTensor<Self>, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2], dilation: [usize; 2], ceil_mode: bool, ) -> MaxPool2dWithIndices<Self> {
        let selection = Selection::tensors(core::iter::once(&x));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::max_pool2d_with_indices(unpack!($mode, $variant, Float, x, $context), kernel_size, stride, padding, dilation, ceil_mode);
                MaxPool2dWithIndices { output: pack!($mode, $variant, Float, result.output), indices: pack!($mode, $variant, Int, result.indices) }
            }};
        }
        dispatch!(selection, call)
    }

    fn max_pool2d_with_indices_backward( x: FloatTensor<Self>, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2], dilation: [usize; 2], ceil_mode: bool, output_grad: FloatTensor<Self>, indices: IntTensor<Self>, ) -> MaxPool2dBackward<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&output_grad)).chain(core::iter::once(&indices)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::max_pool2d_with_indices_backward(unpack!($mode, $variant, Float, x, $context), kernel_size, stride, padding, dilation, ceil_mode, unpack!($mode, $variant, Float, output_grad, $context), unpack!($mode, $variant, Int, indices, $context));
                MaxPool2dBackward { x_grad: pack!($mode, $variant, Float, result.x_grad) }
            }};
        }
        dispatch!(selection, call)
    }

    fn interpolate( x: FloatTensor<Self>, output_size: [usize; 2], options: InterpolateOptions, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::interpolate(unpack!($mode, $variant, Float, x, $context), output_size, options);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn interpolate_backward( x: FloatTensor<Self>, grad: FloatTensor<Self>, output_size: [usize; 2], options: InterpolateOptions, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::interpolate_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, grad, $context), output_size, options);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn interpolate1d(x: FloatTensor<Self>, size: usize, options: InterpolateOptions) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::interpolate1d(unpack!($mode, $variant, Float, x, $context), size, options);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn interpolate1d_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>, size: usize, options: InterpolateOptions) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::interpolate1d_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, grad, $context), size, options);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn interpolate3d(x: FloatTensor<Self>, size: [usize; 3], options: InterpolateOptions) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::interpolate3d(unpack!($mode, $variant, Float, x, $context), size, options);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn interpolate3d_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>, size: [usize; 3], options: InterpolateOptions) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&x).chain(core::iter::once(&grad)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::interpolate3d_backward(unpack!($mode, $variant, Float, x, $context), unpack!($mode, $variant, Float, grad, $context), size, options);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn attention( query: FloatTensor<Self>, key: FloatTensor<Self>, value: FloatTensor<Self>, mask: Option<BoolTensor<Self>>, attn_bias: Option<FloatTensor<Self>>, options: AttentionModuleOptions, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&query).chain(core::iter::once(&key)).chain(core::iter::once(&value)).chain(mask.iter()).chain(attn_bias.iter()));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::attention(unpack!($mode, $variant, Float, query, $context), unpack!($mode, $variant, Float, key, $context), unpack!($mode, $variant, Float, value, $context), mask.map(|value| unpack!($mode, $variant, Bool, value, $context)), attn_bias.map(|value| unpack!($mode, $variant, Float, value, $context)), options);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn layer_norm( tensor: FloatTensor<Self>, gamma: FloatTensor<Self>, beta: Option<FloatTensor<Self>>, epsilon: f64, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&gamma)).chain(beta.iter()));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::layer_norm(unpack!($mode, $variant, Float, tensor, $context), unpack!($mode, $variant, Float, gamma, $context), beta.map(|value| unpack!($mode, $variant, Float, value, $context)), epsilon);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn rms_norm(tensor: FloatTensor<Self>, gamma: FloatTensor<Self>, epsilon: f64) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&gamma)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::rms_norm(unpack!($mode, $variant, Float, tensor, $context), unpack!($mode, $variant, Float, gamma, $context), epsilon);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn rms_norm_with_stats(tensor: FloatTensor<Self>, gamma: FloatTensor<Self>, epsilon: f64) -> RmsNormOutput<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&gamma)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::rms_norm_with_stats(unpack!($mode, $variant, Float, tensor, $context), unpack!($mode, $variant, Float, gamma, $context), epsilon);
                RmsNormOutput { output: pack!($mode, $variant, Float, result.output), rstd: pack!($mode, $variant, Float, result.rstd) }
            }};
        }
        dispatch!(selection, call)
    }

    fn rms_norm_backward(tensor: FloatTensor<Self>, gamma: FloatTensor<Self>, grad: FloatTensor<Self>, rstd: FloatTensor<Self>) -> RmsNormBackward<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&gamma)).chain(core::iter::once(&grad)).chain(core::iter::once(&rstd)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::rms_norm_backward(unpack!($mode, $variant, Float, tensor, $context), unpack!($mode, $variant, Float, gamma, $context), unpack!($mode, $variant, Float, grad, $context), unpack!($mode, $variant, Float, rstd, $context));
                RmsNormBackward { input: pack!($mode, $variant, Float, result.input), weight: pack!($mode, $variant, Float, result.weight) }
            }};
        }
        dispatch!(selection, call)
    }

    fn rms_norm_backward_select(tensor: FloatTensor<Self>, gamma: FloatTensor<Self>, grad: FloatTensor<Self>, rstd: FloatTensor<Self>, mask: [bool; 2]) -> [Option<FloatTensor<Self>>; 2] {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&gamma)).chain(core::iter::once(&grad)).chain(core::iter::once(&rstd)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::rms_norm_backward_select(unpack!($mode, $variant, Float, tensor, $context), unpack!($mode, $variant, Float, gamma, $context), unpack!($mode, $variant, Float, grad, $context), unpack!($mode, $variant, Float, rstd, $context), mask);
                result.map(|value| value.map(|value| pack!($mode, $variant, Float, value)))
            }};
        }
        dispatch!(selection, call)
    }

    fn layer_norm_with_stats( tensor: FloatTensor<Self>, gamma: FloatTensor<Self>, beta: Option<FloatTensor<Self>>, epsilon: f64, ) -> LayerNormOutput<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&gamma)).chain(beta.iter()));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::layer_norm_with_stats(unpack!($mode, $variant, Float, tensor, $context), unpack!($mode, $variant, Float, gamma, $context), beta.map(|value| unpack!($mode, $variant, Float, value, $context)), epsilon);
                LayerNormOutput { output: pack!($mode, $variant, Float, result.output), mean: pack!($mode, $variant, Float, result.mean), rstd: pack!($mode, $variant, Float, result.rstd) }
            }};
        }
        dispatch!(selection, call)
    }

    fn layer_norm_backward( tensor: FloatTensor<Self>, gamma: FloatTensor<Self>, grad: FloatTensor<Self>, mean: FloatTensor<Self>, rstd: FloatTensor<Self>, ) -> LayerNormBackward<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&gamma)).chain(core::iter::once(&grad)).chain(core::iter::once(&mean)).chain(core::iter::once(&rstd)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::layer_norm_backward(unpack!($mode, $variant, Float, tensor, $context), unpack!($mode, $variant, Float, gamma, $context), unpack!($mode, $variant, Float, grad, $context), unpack!($mode, $variant, Float, mean, $context), unpack!($mode, $variant, Float, rstd, $context));
                LayerNormBackward { input: pack!($mode, $variant, Float, result.input), weight: pack!($mode, $variant, Float, result.weight), bias: pack!($mode, $variant, Float, result.bias) }
            }};
        }
        dispatch!(selection, call)
    }

    fn layer_norm_backward_select(tensor: FloatTensor<Self>, gamma: FloatTensor<Self>, grad: FloatTensor<Self>, mean: FloatTensor<Self>, rstd: FloatTensor<Self>, mask: [bool; 3]) -> [Option<FloatTensor<Self>>; 3] {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&gamma)).chain(core::iter::once(&grad)).chain(core::iter::once(&mean)).chain(core::iter::once(&rstd)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::layer_norm_backward_select(unpack!($mode, $variant, Float, tensor, $context), unpack!($mode, $variant, Float, gamma, $context), unpack!($mode, $variant, Float, grad, $context), unpack!($mode, $variant, Float, mean, $context), unpack!($mode, $variant, Float, rstd, $context), mask);
                result.map(|value| value.map(|value| pack!($mode, $variant, Float, value)))
            }};
        }
        dispatch!(selection, call)
    }

    fn layer_norm_default( tensor: FloatTensor<Self>, gamma: FloatTensor<Self>, beta: Option<FloatTensor<Self>>, epsilon: f64, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&gamma)).chain(beta.iter()));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::layer_norm_default(unpack!($mode, $variant, Float, tensor, $context), unpack!($mode, $variant, Float, gamma, $context), beta.map(|value| unpack!($mode, $variant, Float, value, $context)), epsilon);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn ctc_loss( log_probs: FloatTensor<Self>, targets: IntTensor<Self>, input_lengths: IntTensor<Self>, target_lengths: IntTensor<Self>, blank: usize, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&log_probs).chain(core::iter::once(&targets)).chain(core::iter::once(&input_lengths)).chain(core::iter::once(&target_lengths)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::ctc_loss(unpack!($mode, $variant, Float, log_probs, $context), unpack!($mode, $variant, Int, targets, $context), unpack!($mode, $variant, Int, input_lengths, $context), unpack!($mode, $variant, Int, target_lengths, $context), blank);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn ctc_loss_backward( _log_probs: FloatTensor<Self>, _targets: IntTensor<Self>, _input_lengths: IntTensor<Self>, _target_lengths: IntTensor<Self>, _grad_loss: FloatTensor<Self>, _blank: usize, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&_log_probs).chain(core::iter::once(&_targets)).chain(core::iter::once(&_input_lengths)).chain(core::iter::once(&_target_lengths)).chain(core::iter::once(&_grad_loss)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::ctc_loss_backward(unpack!($mode, $variant, Float, _log_probs, $context), unpack!($mode, $variant, Int, _targets, $context), unpack!($mode, $variant, Int, _input_lengths, $context), unpack!($mode, $variant, Int, _target_lengths, $context), unpack!($mode, $variant, Float, _grad_loss, $context), _blank);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn rfft( signal: FloatTensor<Self>, dim: usize, n: Option<usize>, ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let selection = Selection::tensors(core::iter::once(&signal));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::rfft(unpack!($mode, $variant, Float, signal, $context), dim, n);
                { let (value0, value1) = result; (pack!($mode, $variant, Float, value0), pack!($mode, $variant, Float, value1)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn irfft( spectrum_re: FloatTensor<Self>, spectrum_im: FloatTensor<Self>, dim: usize, n: Option<usize>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&spectrum_re).chain(core::iter::once(&spectrum_im)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::irfft(unpack!($mode, $variant, Float, spectrum_re, $context), unpack!($mode, $variant, Float, spectrum_im, $context), dim, n);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }
}
