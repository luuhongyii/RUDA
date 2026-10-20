use crate::{BatchNormConfig, TestBackend as B};
use alloc::vec::Vec;
use ruda_autodiff::{Autodiff, checkpoint::strategy::{BalancedCheckpointing, CheckpointStrategy, NoCheckpointing}};
use ruda_model::tensor::{DType, Tensor, TensorData, Tolerance, module::{batch_norm, batch_norm_train}};
use rstest::rstest;

type AD<C = NoCheckpointing> = Autodiff<B, C>;

fn is_tracked<C: CheckpointStrategy, const D: usize>(tensor: &Tensor<AD<C>, D>) -> bool {
    tensor.clone().into_primitive().tensor().is_tracked()
}

fn fixture() -> [[[f32; 2]; 3]; 2] {
    [
        [[0.9601, 0.7277], [0.6272, 0.9034], [0.9378, 0.7230]],
        [[0.6356, 0.1362], [0.0249, 0.9509], [0.6600, 0.5945]],
    ]
}

#[test]
fn baseline_batch_norm_forward() {
    let device = Default::default();
    let output = batch_norm(
        Tensor::<B, 3>::from_data(fixture(), &device),
        Tensor::from_data([2f32, 3., 4.], &device),
        Tensor::from_data([0.5f32, -1., 2.], &device),
        Tensor::from_data([1f32, 2., 3.], &device),
        Tensor::from_data([3f32, 8., 15.], &device), 1.0,
    );
    let expected = TensorData::from([
        [[0.4601f32, 0.2277], [-2.3728, -2.0966], [-0.0622, -0.2770]],
        [[0.1356, -0.3638], [-2.9751, -2.0491], [-0.3400, -0.4055]],
    ]);
    output.into_data().assert_approx_eq::<f32>(&expected, Tolerance::relative(1e-5));
}

#[test]
fn baseline_training_returns_the_statistics_used_for_normalization() {
    let device = Default::default();
    let input = Tensor::<B, 3>::from_data(fixture(), &device);
    let gamma = Tensor::from_data([2f32, 3., 4.], &device);
    let beta = Tensor::from_data([0.5f32, -1., 2.], &device);
    let result = batch_norm_train(input.clone(), gamma.clone(), beta.clone(), 1e-5);
    let mean = input.clone().swap_dims(0, 1).reshape([3, 4]).mean_dim(1).reshape([3]);
    let variance = input.clone().sub(mean.clone().reshape([1, 3, 1])).square()
        .swap_dims(0, 1).reshape([3, 4]).mean_dim(1).reshape([3]);
    let expected = batch_norm(input, gamma, beta, mean.clone(), variance.clone(), 1e-5);
    let tolerance = Tolerance::relative(1e-4);
    result.mean.into_data().assert_approx_eq::<f32>(&mean.into_data(), tolerance);
    result.variance.into_data().assert_approx_eq::<f32>(&variance.into_data(), tolerance);
    result.output.into_data().assert_approx_eq::<f32>(&expected.into_data(), tolerance);
}

#[test]
fn baseline_large_batch_statistics_f32_and_f16() {
    let device = Default::default();
    for dtype in [DType::F32, DType::F16] {
        for batch in [2, 8] {
            let shape = [batch, 2, 128, 128];
            let values: Vec<f32> = (0..batch * 2 * 128 * 128)
                .map(|i| if (i / (128 * 128)) % 2 == 0 || i % 2 == 0 { 3.0 } else { -3.0 }).collect();
            let expected: Vec<f32> = (0..values.len())
                .map(|i| if (i / (128 * 128)) % 2 == 0 { 0.0 } else { values[i] / 3.0 }).collect();
            let input = Tensor::<B, 4>::from_data(TensorData::new(values, shape), (&device, dtype));
            let result = batch_norm_train(input, Tensor::ones([2], (&device, dtype)),
                Tensor::zeros([2], (&device, dtype)), 1e-5);
            let tolerance = Tolerance::absolute(1e-3);
            result.mean.into_data().assert_approx_eq::<f32>(&TensorData::from([3f32, 0.]), tolerance);
            result.variance.into_data().assert_approx_eq::<f32>(&TensorData::from([0f32, 9.]), tolerance);
            result.output.into_data().assert_approx_eq::<f32>(&TensorData::new(expected, shape), tolerance);
        }
    }
}

fn baseline_gradients<C: CheckpointStrategy>() {
    let device = Default::default();
    let input_data = [
        [[0.9601f32, 0.7277, 0.1], [0.6272, 0.9034, 0.4], [0.9378, 0.7230, 0.2]],
        [[0.6356, 0.1362, 0.7], [0.0249, 0.9509, 0.8], [0.6600, 0.5945, 0.3]],
    ];
    let weights = Tensor::<AD<C>, 3>::from_data([
        [[1f32, -2., 0.5], [0.3, 0.7, -1.1], [2., 0.1, 0.4]],
        [[-0.6, 1.4, 0.9], [1.2, -0.8, 0.2], [0.5, 0.5, -1.5]],
    ], &device);
    let x = Tensor::<AD<C>, 3>::from_data(input_data, &device).require_grad();
    let gamma = Tensor::from_data([2f32, 3., 4.], &device).require_grad();
    let beta = Tensor::from_data([0.5f32, -1., 2.], &device).require_grad();
    let result = batch_norm_train(x.clone(), gamma.clone(), beta.clone(), 1e-5);
    assert!(is_tracked(&result.output));
    assert_eq!(result.mean.device(), device);
    assert_eq!(result.variance.device(), device);
    assert!(!is_tracked(&result.mean));
    assert!(!is_tracked(&result.variance));
    let grads = result.output.mul(weights.clone()).sum().backward();

    let x_ref = Tensor::<AD<C>, 3>::from_data(input_data, &device).require_grad();
    let gamma_ref = Tensor::from_data([2f32, 3., 4.], &device).require_grad();
    let beta_ref = Tensor::from_data([0.5f32, -1., 2.], &device).require_grad();
    let mean = x_ref.clone().swap_dims(0, 1).reshape([3, 6]).mean_dim(1).reshape([1, 3, 1]);
    let variance = x_ref.clone().sub(mean.clone()).square().swap_dims(0, 1)
        .reshape([3, 6]).mean_dim(1).reshape([1, 3, 1]);
    let output = batch_norm(x_ref.clone(), gamma_ref.clone(), beta_ref.clone(),
        mean.reshape([3]), variance.reshape([3]), 1e-5);
    let reference = output.mul(weights).sum().backward();
    let tolerance = Tolerance::relative(1e-4);
    x.grad(&grads).unwrap().into_data().assert_approx_eq::<f32>(&x_ref.grad(&reference).unwrap().into_data(), tolerance);
    gamma.grad(&grads).unwrap().into_data().assert_approx_eq::<f32>(&gamma_ref.grad(&reference).unwrap().into_data(), tolerance);
    beta.grad(&grads).unwrap().into_data().assert_approx_eq::<f32>(&beta_ref.grad(&reference).unwrap().into_data(), tolerance);
}

fn baseline_large_f16_gradient<C: CheckpointStrategy>() {
    let device = Default::default();
    for batch in [2, 8] {
        let shape = [batch, 1, 128, 128];
        let values: Vec<f32> = (0..batch * 128 * 128).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let expected = TensorData::new(values.iter().map(|x| x * 0.75).collect::<Vec<_>>(), shape);
        let x = Tensor::<AD<C>, 4>::from_data(TensorData::new(values, shape), (&device, DType::F16)).require_grad();
        let gamma = Tensor::full([1], 2.0, (&device, DType::F16));
        let beta = Tensor::zeros([1], (&device, DType::F16));
        let weights = x.clone().detach().add_scalar(2.0);
        let result = batch_norm_train(x.clone(), gamma, beta, 3.0);
        let grads = result.output.mul(weights).sum().backward();
        x.grad(&grads).unwrap().into_data().assert_approx_eq::<f32>(&expected, Tolerance::absolute(1e-3));
    }
}

fn exact_layout<C: CheckpointStrategy, const D: usize>(shape: [usize; D], dtype: DType, strided: bool, tracked: u8) {
    let device = Default::default();
    let mut storage = shape;
    if strided { storage.swap(0, D - 1); }
    let count: usize = shape.iter().product();
    let values: Vec<f32> = (0..count).map(|i| if i < count / 2 { 1.0 } else { -1.0 }).collect();
    let mut x = Tensor::<AD<C>, D>::from_data(TensorData::new(values, storage), (&device, dtype));
    if strided { x = x.swap_dims(0, D - 1); }
    let x = x.set_require_grad(tracked & 1 != 0);
    let gamma = Tensor::<AD<C>, 1>::from_data([2f32, -4.], (&device, dtype)).set_require_grad(tracked & 2 != 0);
    let beta = Tensor::<AD<C>, 1>::from_data([0.5f32, -1.], (&device, dtype)).set_require_grad(tracked & 4 != 0);
    let values = x.to_data().try_into_vec_as::<f32>().unwrap();
    let spatial: usize = shape[2..].iter().product();
    let expected: Vec<f32> = values.iter().enumerate().map(|(i, x)| {
        let c = (i / spatial) % 2;
        x * [1., -2.][c] + [0.5, -1.][c]
    }).collect();
    let result = batch_norm_train(x.clone().add_scalar(1.0).sub_scalar(1.0), gamma.clone(), beta.clone(), 3.0);
    assert_eq!(result.output.dtype(), dtype);
    assert_eq!(result.mean.dtype(), dtype);
    assert_eq!(result.variance.dtype(), dtype);
    result.mean.to_data().assert_eq(&TensorData::from([0f32, 0.]).convert_dtype(dtype), true);
    result.variance.to_data().assert_eq(&TensorData::from([1f32, 1.]).convert_dtype(dtype), true);
    result.output.to_data().assert_eq(&TensorData::new(expected, shape).convert_dtype(dtype), true);
    assert!(!is_tracked(&result.mean) && !is_tracked(&result.variance));
    assert_eq!(is_tracked(&result.output), tracked != 0);
    if tracked == 0 { return; }
    let gradients = result.output.mul(x.clone().detach().add_scalar(2.0)).sum().backward();
    assert_eq!(x.grad(&gradients).is_some(), tracked & 1 != 0);
    assert_eq!(gamma.grad(&gradients).is_some(), tracked & 2 != 0);
    assert_eq!(beta.grad(&gradients).is_some(), tracked & 4 != 0);
    if let Some(grad) = x.grad(&gradients) {
        let expected: Vec<f32> = values.iter().enumerate().map(|(i, x)| x * [0.75, -1.5][(i / spatial) % 2]).collect();
        grad.into_data().assert_eq(&TensorData::new(expected, shape).convert_dtype(dtype), true);
    }
    if let Some(grad) = gamma.grad(&gradients) {
        grad.into_data().assert_eq(&TensorData::from([count as f32 / 4.; 2]).convert_dtype(dtype), true);
    }
    if let Some(grad) = beta.grad(&gradients) {
        grad.into_data().assert_eq(&TensorData::from([count as f32; 2]).convert_dtype(dtype), true);
    }
}

macro_rules! gradient_cases {
    ($strategy:ty) => {
        #[test]
        fn baseline_gradients_match_decomposed_statistics() { super::baseline_gradients::<$strategy>(); }
        #[test]
        fn baseline_large_f16_input_gradient() { super::baseline_large_f16_gradient::<$strategy>(); }
        #[test]
        fn rank_dtype_layout_and_independent_gradient_leaves() {
            super::exact_layout::<$strategy, 2>([2, 2], DType::F32, false, 7);
            super::exact_layout::<$strategy, 3>([2, 2, 3], DType::F32, false, 7);
            super::exact_layout::<$strategy, 5>([2, 2, 2, 2, 2], DType::F32, false, 7);
            for dtype in [DType::F32, DType::F64, DType::F16, DType::BF16] {
                super::exact_layout::<$strategy, 4>([2, 2, 2, 2], dtype, true, 7);
            }
            for tracked in 0..8 {
                super::exact_layout::<$strategy, 3>([2, 2, 3], DType::F32, false, tracked);
            }
        }
        #[test]
        fn layer_momentum_boundaries() {
            let device = Default::default();
            for (momentum, mean, variance) in [(0.0, 0f32, 1f32), (1.0, 4., 4.)] {
                let layer = BatchNormConfig::new(1).with_momentum(momentum).init::<AD<$strategy>>(&device);
                layer.forward(Tensor::<_, 2>::from_data([[1f32], [3.]], &device));
                let output = layer.forward(Tensor::<_, 2>::from_data([[2f32], [6.]], &device));
                output.into_data().assert_approx_eq::<f32>(&TensorData::from([[-1f32], [1.]]), Tolerance::default());
                layer.running_mean.value_sync().into_data().assert_approx_eq::<f32>(&TensorData::from([mean]), Tolerance::default());
                layer.running_var.value_sync().into_data().assert_approx_eq::<f32>(&TensorData::from([variance]), Tolerance::default());
                assert!(!is_tracked(&layer.running_mean.value_sync()));
                assert!(!is_tracked(&layer.running_var.value_sync()));
            }
        }
    };
}

mod plain { use super::*; gradient_cases!(NoCheckpointing); }
mod balanced { use super::*; gradient_cases!(BalancedCheckpointing); }

#[test]
#[should_panic(expected = "num_features must be greater than zero.")]
fn zero_features_rejected() { BatchNormConfig::new(0).init::<B>(&Default::default()); }

#[rstest]
#[case(0.0)]
#[case(-1e-5)]
#[case(f64::NAN)]
#[case(f64::INFINITY)]
#[case(f64::NEG_INFINITY)]
#[should_panic(expected = "epsilon must be finite and positive.")]
fn invalid_epsilon_rejected(#[case] epsilon: f64) {
    BatchNormConfig::new(1).with_epsilon(epsilon).init::<B>(&Default::default());
}

#[rstest]
#[case(-0.1)]
#[case(1.1)]
#[case(f64::NAN)]
#[case(f64::INFINITY)]
#[case(f64::NEG_INFINITY)]
#[should_panic(expected = "momentum must be finite and within [0, 1].")]
fn invalid_momentum_rejected(#[case] momentum: f64) {
    BatchNormConfig::new(1).with_momentum(momentum).init::<B>(&Default::default());
}

#[test]
#[should_panic(expected = "rank >= 2")]
fn layer_rejects_rank_one() {
    let device = Default::default();
    BatchNormConfig::new(3).init::<B>(&device).forward(Tensor::<B, 1>::zeros([4], &device));
}

#[test]
#[should_panic(expected = "BatchNorm input channels must match num_features")]
fn layer_rejects_wrong_channels() {
    let device = Default::default();
    BatchNormConfig::new(3).init::<B>(&device).forward(Tensor::<B, 3>::zeros([1, 4, 2], &device));
}
