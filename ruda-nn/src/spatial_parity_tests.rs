use crate::{
    interpolate::{
        Interpolate1dConfig, Interpolate2dConfig, Interpolate3dConfig, InterpolateMode as LayerMode,
    },
    loss::{LpLossConfig, SmoothL1LossConfig},
};
use alloc::{vec, vec::Vec};
use ruda_autodiff::Autodiff;
use ruda_model::tensor::{
    DType, Tensor, TensorData, Tolerance,
    backend::Backend,
    module,
    ops::{InterpolateMode, InterpolateOptions},
};

fn coordinates<const D: usize>(mut flat: usize, shape: [usize; D]) -> [usize; D] {
    let mut result = [0; D];
    for axis in (0..D).rev() {
        result[axis] = flat % shape[axis];
        flat /= shape[axis];
    }
    result
}

fn linear<const D: usize>(coordinate: [usize; D], shape: [usize; D]) -> usize {
    coordinate
        .into_iter()
        .zip(shape)
        .fold(0, |offset, (index, size)| offset * size + index)
}

fn check_resize<B: Backend, const D: usize>(
    input_shape: [usize; D],
    output_shape: [usize; D],
    dtype: DType,
    strided: bool,
    exact: bool,
    resize: impl Fn(Tensor<Autodiff<B>, D>) -> Tensor<Autodiff<B>, D>,
) {
    let device = B::Device::default();
    let mut storage_shape = input_shape;
    if strided {
        storage_shape.swap(D - 2, D - 1);
    }
    let values: Vec<f64> = (0..storage_shape.iter().product())
        .map(|i| {
            (i % 31) as f64 / 8.0 - 1.0
                + if dtype == DType::F64 {
                    2f64.powi(-40)
                } else {
                    0.0
                }
        })
        .collect();
    let mut input = Tensor::<Autodiff<B>, D>::from_data(
        TensorData::new(values.clone(), storage_shape),
        (&device, dtype),
    );
    if strided {
        input = input.swap_dims(D - 2, D - 1);
    }
    let input = input.require_grad();
    let output = resize(input.clone());
    assert_eq!(output.dims(), output_shape);
    assert_eq!(output.dtype(), dtype);
    let weights: Vec<f64> = (0..output_shape.iter().product())
        .map(|i| (i % 5) as f64 - 2.0)
        .collect();
    let mut expected = Vec::new();
    let mut gradient = vec![0.0f64; input_shape.iter().product()];
    for (flat, weight) in weights.iter().enumerate() {
        let mut source = coordinates(flat, output_shape);
        for axis in 2..D {
            let numerator = if exact {
                (2 * source[axis] as u128 + 1) * input_shape[axis] as u128
            } else {
                source[axis] as u128 * input_shape[axis] as u128
            };
            let denominator = output_shape[axis] as u128 * if exact { 2 } else { 1 };
            source[axis] = (numerator / denominator) as usize;
        }
        gradient[linear(source, input_shape)] += weight;
        if strided {
            source.swap(D - 2, D - 1);
        }
        expected.push(values[linear(source, storage_shape)]);
    }
    assert_eq!(output.to_data().iter::<f64>().collect::<Vec<_>>(), expected);
    let weights = Tensor::from_data(TensorData::new(weights, output_shape), (&device, dtype));
    let gradients = (output * weights).sum().backward();
    let actual = input.grad(&gradients).unwrap();
    assert_eq!(actual.dims(), input_shape);
    assert_eq!(actual.dtype(), dtype);
    assert_eq!(
        actual.into_data().iter::<f64>().collect::<Vec<_>>(),
        gradient
    );
}

fn image_cases<B: Backend>(dtype: DType, strided: bool, exact: bool) {
    for (shape, size) in [
        ([1, 1, 2, 2], [3, 3]),
        ([2, 3, 3, 4], [5, 2]),
        ([1, 8, 2, 3], [4, 7]),
        ([1, 2, 7, 5], [3, 2]),
        ([2, 1, 1, 1], [3, 4]),
        ([1, 4, 6, 7], [1, 1]),
    ] {
        for align_corners in [false, true] {
            let mode = if exact {
                InterpolateMode::NearestExact
            } else {
                InterpolateMode::Nearest
            };
            let options = InterpolateOptions::new(mode).with_align_corners(align_corners);
            check_resize::<B, 4>(
                shape,
                [shape[0], shape[1], size[0], size[1]],
                dtype,
                strided,
                exact,
                |x| module::interpolate(x, size, options.clone()),
            );
        }
    }
}

fn line_cases<B: Backend>(dtype: DType) {
    for (length, size) in [(2, 3), (7, 3), (8, 1), (1, 7)] {
        check_resize::<B, 3>([2, 3, length], [2, 3, size], dtype, false, true, |x| {
            module::interpolate1d(
                x,
                size,
                InterpolateOptions::new(InterpolateMode::NearestExact),
            )
        });
    }
}

fn volume_case<B: Backend>(dtype: DType, strided: bool) {
    check_resize::<B, 5>(
        [2, 3, 3, 2, 4],
        [2, 3, 2, 4, 3],
        dtype,
        strided,
        true,
        |x| {
            module::interpolate3d(
                x,
                [2, 4, 3],
                InterpolateOptions::new(InterpolateMode::NearestExact),
            )
        },
    );
}

fn layer_cases<B: Backend>() {
    let line = Interpolate1dConfig::new()
        .with_scale_factor(Some(1.5))
        .with_mode(LayerMode::NearestExact)
        .init();
    check_resize::<B, 3>([2, 3, 5], [2, 3, 7], DType::F32, false, true, |x| {
        line.forward(x)
    });
    check_resize::<B, 3>([2, 3, 5], [2, 3, 3], DType::F32, false, true, |x| {
        line.forward_with_output_size(x, 3)
    });
    let image = Interpolate2dConfig::new()
        .with_scale_factor(Some([0.75, 1.5]))
        .with_mode(LayerMode::NearestExact)
        .init();
    check_resize::<B, 4>([2, 3, 7, 5], [2, 3, 5, 7], DType::F32, true, true, |x| {
        image.forward(x)
    });
    check_resize::<B, 4>([2, 3, 7, 5], [2, 3, 2, 3], DType::F32, true, true, |x| {
        image.forward_with_output_size(x, [2, 3])
    });
    let volume = Interpolate3dConfig::new()
        .with_output_size(Some([2, 4, 3]))
        .with_mode(LayerMode::NearestExact)
        .init();
    check_resize::<B, 5>(
        [2, 3, 3, 2, 4],
        [2, 3, 2, 4, 3],
        DType::F32,
        false,
        true,
        |x| volume.forward(x),
    );
}

fn loss_axes<B: Backend>(smooth: bool, parameter: f64) {
    let device = B::Device::default();
    let values: Vec<f32> = (0..24)
        .map(|i| ((i % 7 + 1) as f32 / 4.0) * if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let prediction =
        Tensor::<Autodiff<B>, 3>::from_data(TensorData::new(values.clone(), [2, 3, 4]), &device)
            .require_grad();
    let target = Tensor::<Autodiff<B>, 3>::zeros([2, 3, 4], &device).require_grad();
    let mut expected = vec![0.0f32; 3];
    let mut gradient = Vec::new();
    for (i, &x) in values.iter().enumerate() {
        let x = x as f64;
        let (value, derivative) = if smooth {
            if x.abs() < parameter {
                (0.5 * x * x / parameter, x / parameter)
            } else {
                (x.abs() - 0.5 * parameter, x.signum())
            }
        } else {
            (
                x.abs().powf(parameter),
                parameter * x.signum() * x.abs().powf(parameter - 1.0),
            )
        };
        expected[(i / 4) % 3] += value as f32 / 8.0;
        gradient.push(derivative as f32 / 8.0);
    }
    let output = if smooth {
        SmoothL1LossConfig::new()
            .with_beta(parameter as f32)
            .init()
            .forward_reduce_axes(prediction.clone(), target.clone(), &[-1i32, -3])
    } else {
        LpLossConfig::new(parameter).init().forward_reduce_axes(
            prediction.clone(),
            target.clone(),
            &[-1i64, -3],
        )
    };
    assert_eq!(output.dims(), [1, 3, 1]);
    output.to_data().assert_approx_eq::<f32>(
        &TensorData::new(expected, [1, 3, 1]),
        Tolerance::rel_abs(3e-5, 3e-6),
    );
    let grads = output.sum().backward();
    prediction
        .grad(&grads)
        .unwrap()
        .into_data()
        .assert_approx_eq::<f32>(
            &TensorData::new(gradient.clone(), [2, 3, 4]),
            Tolerance::rel_abs(3e-5, 3e-6),
        );
    target
        .grad(&grads)
        .unwrap()
        .into_data()
        .assert_approx_eq::<f32>(
            &TensorData::new(
                gradient.into_iter().map(|x| -x).collect::<Vec<_>>(),
                [2, 3, 4],
            ),
            Tolerance::rel_abs(3e-5, 3e-6),
        );
}

fn axis_compatibility<B: Backend>() {
    let device = B::Device::default();
    let x = Tensor::<B, 3>::from_data(
        TensorData::new(
            (1..=24).map(|i| i as f32 / 4.0).collect::<Vec<_>>(),
            [2, 3, 4],
        ),
        &device,
    );
    let z = Tensor::<B, 3>::zeros([2, 3, 4], &device);
    let lp = LpLossConfig::l2();
    let smooth = SmoothL1LossConfig::new().init();
    for (old, signed, empty_old, empty_signed) in [
        (
            lp.forward_reduce_dims(x.clone(), z.clone(), &[2, 0]),
            lp.forward_reduce_axes(x.clone(), z.clone(), &[-1isize, 2, -3]),
            lp.forward_reduce_dims(x.clone(), z.clone(), &[]),
            lp.forward_reduce_axes(x.clone(), z.clone(), &[] as &[i32]),
        ),
        (
            smooth.forward_reduce_dims(x.clone(), z.clone(), &[2, 0]),
            smooth.forward_reduce_axes(x.clone(), z.clone(), &[-1isize, 2, -3]),
            smooth.forward_reduce_dims(x.clone(), z.clone(), &[]),
            smooth.forward_reduce_axes(x, z, &[] as &[i32]),
        ),
    ] {
        old.into_data().assert_eq(&signed.into_data(), false);
        empty_old
            .into_data()
            .assert_eq(&empty_signed.into_data(), false);
    }
}

macro_rules! cases {
    ($backend:ty, $fusion:expr) => {
        type B = $backend;
        #[test]
        fn backend_identity() {
            let name = B::name(&Default::default());
            assert_eq!(name.starts_with("fusion<"), $fusion, "{name}");
        }
        #[test]
        fn nearest_exact_2d_f32() {
            image_cases::<B>(DType::F32, false, true);
        }
        #[test]
        fn nearest_exact_2d_strided() {
            image_cases::<B>(DType::F32, true, true);
        }
        #[test]
        fn nearest_exact_2d_f64() {
            image_cases::<B>(DType::F64, false, true);
        }
        #[test]
        fn nearest_exact_2d_f16() {
            image_cases::<B>(DType::F16, false, true);
        }
        #[test]
        fn nearest_exact_2d_bf16() {
            image_cases::<B>(DType::BF16, false, true);
        }
        #[test]
        fn nearest_floor_unchanged() {
            image_cases::<B>(DType::F32, false, false);
        }
        #[test]
        fn nearest_exact_1d_f32() {
            line_cases::<B>(DType::F32);
        }
        #[test]
        fn nearest_exact_1d_f16_bf16() {
            for dtype in [DType::F16, DType::BF16] {
                line_cases::<B>(dtype);
            }
        }
        #[test]
        fn nearest_exact_3d_f32() {
            volume_case::<B>(DType::F32, false);
        }
        #[test]
        fn nearest_exact_3d_strided() {
            volume_case::<B>(DType::F32, true);
        }
        #[test]
        fn nearest_exact_3d_f16_bf16() {
            for dtype in [DType::F16, DType::BF16] {
                volume_case::<B>(dtype, false);
            }
        }
        #[test]
        fn nearest_exact_nn_layers() {
            layer_cases::<B>();
        }
        #[cfg(feature = "ruda-tensor-router")]
        #[test]
        fn nearest_exact_router() {
            layer_cases::<ruda_tensor_router::Router<(B, B)>>();
        }
        #[test]
        fn lp_signed_axes_forward_backward() {
            for p in [0.5, 1.0, 2.0, 3.0] {
                loss_axes::<B>(false, p);
            }
        }
        #[test]
        fn smooth_l1_signed_axes_forward_backward() {
            for beta in [0.5, 2.0] {
                loss_axes::<B>(true, beta);
            }
        }
        #[test]
        fn signed_axes_preserve_legacy_and_empty_calls() {
            axis_compatibility::<B>();
        }
        #[test]
        #[should_panic]
        fn lp_negative_axis_out_of_bounds() {
            let device = Default::default();
            let x = Tensor::<B, 3>::ones([1, 2, 3], &device);
            LpLossConfig::l2().forward_reduce_axes(x.clone(), x, &[-4]);
        }
        #[test]
        #[should_panic]
        fn smooth_l1_positive_axis_out_of_bounds() {
            let device = Default::default();
            let x = Tensor::<B, 3>::ones([1, 2, 3], &device);
            SmoothL1LossConfig::new()
                .init()
                .forward_reduce_axes(x.clone(), x, &[3]);
        }
    };
}

#[cfg(not(any(feature = "test-tch", feature = "test-wgpu", feature = "test-rocm")))]
mod selected {
    use super::*;
    cases!(
        crate::TestBackend,
        cfg!(all(feature = "test-cuda", not(feature = "test-cuda-native")))
    );
}

#[cfg(feature = "test-cuda")]
mod native {
    use super::*;
    cases!(ruda_tensor_device::cuda::CudaNative, false);
}

#[test]
fn nearest_exact_ir_round_trip() {
    use ruda_tensor::graph::{InterpolateModeIr, InterpolateOptionsIr};
    for align in [false, true] {
        let original =
            InterpolateOptions::new(InterpolateMode::NearestExact).with_align_corners(align);
        let ir: InterpolateOptionsIr = original.into();
        assert!(matches!(ir.mode, InterpolateModeIr::NearestExact));
        #[cfg(feature = "serde_json")]
        let ir = serde_json::from_str::<InterpolateOptionsIr>(&serde_json::to_string(&ir).unwrap())
            .unwrap();
        let options: InterpolateOptions = ir.into();
        assert!(matches!(options.mode, InterpolateMode::NearestExact));
        assert_eq!(options.align_corners, align);
    }
}
