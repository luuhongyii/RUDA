use super::*;
use ruda_tensor::api::{FloatDType, Tensor, TensorData, Tolerance};

type B = DeviceBackend<CudaRuntime, f32, i32, u8>;

fn precision(dtype: FloatDType, scale: f64, tolerance: f64) {
    let device = Default::default();
    let lhs = Tensor::<B, 1>::from_data(
        TensorData::from([scale, -3., 0., scale.recip()]),
        (&device, dtype.into()),
    );
    let rhs = Tensor::<B, 1>::from_data(
        TensorData::from([scale, 4., -0., scale.recip()]),
        (&device, dtype.into()),
    );
    let a = lhs.to_data().convert::<f64>().to_vec::<f64>().unwrap();
    let b = rhs.to_data().convert::<f64>().to_vec::<f64>().unwrap();
    assert!(a[0].is_finite() && a[3] > 0. && b[0].is_finite() && b[3] > 0.);
    let out = lhs.hypot(rhs);
    assert_eq!(out.dtype(), dtype.into());
    let values = out.into_data().convert::<f64>().to_vec::<f64>().unwrap();
    for ((actual, a), b) in values.into_iter().zip(a).zip(b) {
        let expected = a.hypot(b);
        assert!(actual.is_finite());
        assert!(
            (actual - expected).abs() <= expected.abs() * tolerance,
            "{dtype:?}: {actual} != {expected}"
        );
        if expected == 0. {
            assert_eq!(actual.to_bits(), 0);
        }
    }
}

#[test]
fn native_f16() {
    precision(FloatDType::F16, 10000., 0.001);
}

#[test]
fn native_bf16() {
    precision(FloatDType::BF16, 1e30, 0.01);
}

#[test]
fn native_f32() {
    precision(FloatDType::F32, 1e30, 3e-7);
}

#[test]
fn native_f64() {
    precision(FloatDType::F64, 1e300, 3e-15);
}

#[test]
fn native_strided_broadcast_and_empty() {
    let device = Default::default();
    let lhs = Tensor::<B, 2>::from_floats([[3., 5., 8.], [-3., -5., -8.]], &device).transpose();
    let rhs = Tensor::<B, 2>::from_floats([[4.], [12.], [15.]], &device);
    lhs.hypot(rhs).into_data().assert_approx_eq::<f32>(
        &TensorData::from([[5., 5.], [13., 13.], [17., 17.]]),
        Tolerance::default(),
    );
    let empty = Tensor::<B, 2>::from_data(TensorData::new(Vec::<f32>::new(), [0, 3]), &device);
    let out = empty.hypot(Tensor::ones([1, 3], &device));
    assert_eq!(out.dims(), [0, 3]);
    assert_eq!(out.into_data().num_elements(), 0);
}

#[test]
fn native_nonfinite() {
    let device = Default::default();
    let lhs = Tensor::<B, 1>::from_floats([f32::INFINITY, f32::NAN, f32::NAN], &device);
    let rhs = Tensor::from_floats([f32::NAN, f32::NEG_INFINITY, 1.], &device);
    let out = lhs.hypot(rhs).into_data().to_vec::<f32>().unwrap();
    assert_eq!(&out[..2], &[f32::INFINITY, f32::INFINITY]);
    assert!(out[2].is_nan());
}
