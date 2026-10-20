use crate::{TestAutodiffBackend, TestBackend};
use alloc::vec::Vec;
use ruda_model::tensor::{FloatDType, Tensor, TensorData, Tolerance};

fn close<const D: usize>(actual: Tensor<TestBackend, D>, expected: impl Into<TensorData>) {
    actual
        .into_data()
        .assert_approx_eq::<f32>(&expected.into(), Tolerance::rel_abs(2e-5, 2e-6));
}

#[test]
fn basic() {
    let device = Default::default();
    let lhs = Tensor::<TestBackend, 2>::from_floats([[3., 4.], [5., 12.]], &device);
    let rhs = Tensor::from_floats([[4., 3.], [12., 5.]], &device);
    close(lhs.hypot(rhs), [[5., 5.], [13., 13.]]);
}

#[test]
fn broadcast() {
    let device = Default::default();
    let lhs = Tensor::<TestBackend, 2>::from_floats([[3., 4., 5.]], &device);
    let rhs = Tensor::from_floats([[4.], [3.]], &device);
    close(
        lhs.hypot(rhs),
        [
            [5., 32f32.sqrt(), 41f32.sqrt()],
            [18f32.sqrt(), 5., 34f32.sqrt()],
        ],
    );
}

#[test]
fn strided_and_alias() {
    let device = Default::default();
    let lhs = Tensor::<TestBackend, 2>::from_floats([[3., -5., 8.], [-4., 12., -15.]], &device)
        .transpose();
    let saved = lhs.clone();
    let rhs = Tensor::from_floats([[4., 3.], [12., 5.], [15., 8.]], &device);
    close(lhs.hypot(rhs), [[5., 5.], [13., 13.], [17., 17.]]);
    close(saved, [[3., -4.], [-5., 12.], [8., -15.]]);
}

fn precision(dtype: FloatDType, large: f64, small: f64, tolerance: f64) {
    let device = Default::default();
    let lhs = Tensor::<TestBackend, 1>::from_data(
        TensorData::from([large, small, -3., 0.]),
        (&device, dtype.into()),
    );
    let rhs = Tensor::<TestBackend, 1>::from_data(
        TensorData::from([large, small, 4., -0.]),
        (&device, dtype.into()),
    );
    let a = lhs.to_data().convert::<f64>().to_vec::<f64>().unwrap();
    let b = rhs.to_data().convert::<f64>().to_vec::<f64>().unwrap();
    assert!(a[0].is_finite() && a[1] > 0. && b[0].is_finite() && b[1] > 0.);
    let result = lhs.hypot(rhs);
    assert_eq!(result.dtype(), dtype.into());
    let values = result.into_data().convert::<f64>().to_vec::<f64>().unwrap();
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
fn f16_range() {
    precision(FloatDType::F16, 30000., 0.0001, 0.001);
}

#[test]
fn bf16_range() {
    precision(FloatDType::BF16, 1e30, 1e-30, 0.01);
}

#[test]
fn f32_range() {
    precision(FloatDType::F32, 1e30, 1e-30, 3e-7);
}

#[test]
fn f64_range() {
    precision(FloatDType::F64, 1e300, 1e-300, 3e-15);
}

#[test]
fn nonfinite_and_signed_zero() {
    let device = Default::default();
    let lhs = Tensor::<TestBackend, 1>::from_floats(
        [f32::INFINITY, f32::NAN, f32::NAN, -0., 0., -3.],
        &device,
    );
    let rhs = Tensor::from_floats([f32::NAN, f32::NEG_INFINITY, 2., -0., 4., 0.], &device);
    let out = lhs.hypot(rhs).into_data().to_vec::<f32>().unwrap();
    assert_eq!(&out[..2], &[f32::INFINITY, f32::INFINITY]);
    assert!(out[2].is_nan());
    assert_eq!(out[3].to_bits(), 0);
    assert_eq!(&out[4..], &[4., 3.]);
}

#[test]
fn empty_broadcast() {
    let device = Default::default();
    let lhs =
        Tensor::<TestBackend, 2>::from_data(TensorData::new(Vec::<f32>::new(), [0, 3]), &device);
    let out = lhs.hypot(Tensor::ones([1, 3], &device));
    assert_eq!(out.dims(), [0, 3]);
    assert_eq!(out.into_data().num_elements(), 0);
}

#[test]
#[should_panic(expected = "incompatible shapes")]
fn incompatible_shapes() {
    let device = Default::default();
    Tensor::<TestBackend, 2>::ones([2, 2], &device).hypot(Tensor::ones([2, 3], &device));
}

#[test]
fn backward_both() {
    let device = Default::default();
    let lhs = Tensor::<TestAutodiffBackend, 2>::from_floats([[3., 4.], [5., 12.]], &device)
        .require_grad();
    let rhs = Tensor::<TestAutodiffBackend, 2>::from_floats([[4., 3.], [12., 5.]], &device)
        .require_grad();
    let grads = lhs.clone().hypot(rhs.clone()).backward();
    close(
        lhs.grad(&grads).unwrap(),
        [[0.6, 0.8], [5. / 13., 12. / 13.]],
    );
    close(
        rhs.grad(&grads).unwrap(),
        [[0.8, 0.6], [12. / 13., 5. / 13.]],
    );
}

#[test]
fn backward_broadcast() {
    let device = Default::default();
    let lhs = Tensor::<TestAutodiffBackend, 2>::from_floats([[3., -3.]], &device).require_grad();
    let rhs = Tensor::<TestAutodiffBackend, 2>::from_floats([[4.], [-4.]], &device).require_grad();
    let grads = lhs.clone().hypot(rhs.clone()).sum().backward();
    close(lhs.grad(&grads).unwrap(), [[1.2, -1.2]]);
    close(rhs.grad(&grads).unwrap(), [[1.6], [-1.6]]);
}

#[test]
fn backward_single_parent_and_shared_input() {
    let device = Default::default();
    for track_left in [true, false] {
        let lhs = Tensor::<TestAutodiffBackend, 1>::from_floats([3.], &device)
            .set_require_grad(track_left);
        let rhs = Tensor::<TestAutodiffBackend, 1>::from_floats([4.], &device)
            .set_require_grad(!track_left);
        let grads = lhs.clone().hypot(rhs.clone()).backward();
        if track_left {
            close(lhs.grad(&grads).unwrap(), [0.6]);
            assert!(rhs.grad(&grads).is_none());
        } else {
            close(rhs.grad(&grads).unwrap(), [0.8]);
            assert!(lhs.grad(&grads).is_none());
        }
    }
    let lhs = Tensor::<TestAutodiffBackend, 1>::from_floats([-3., 4.], &device).require_grad();
    let grads = lhs.clone().hypot(lhs.clone()).sum().backward();
    close(lhs.grad(&grads).unwrap(), [-2f32.sqrt(), 2f32.sqrt()]);
}

#[test]
fn backward_origin_is_undefined() {
    let device = Default::default();
    let lhs = Tensor::<TestAutodiffBackend, 1>::zeros([1], &device).require_grad();
    let rhs = Tensor::<TestAutodiffBackend, 1>::zeros([1], &device).require_grad();
    let grads = lhs.clone().hypot(rhs.clone()).backward();
    for grad in [lhs.grad(&grads).unwrap(), rhs.grad(&grads).unwrap()] {
        assert!(grad.into_data().to_vec::<f32>().unwrap()[0].is_nan());
    }
}

#[test]
fn fused_chain_backward() {
    let device = Default::default();
    let lhs = Tensor::<TestAutodiffBackend, 1>::from_floats([3., -3.], &device).require_grad();
    let rhs = Tensor::<TestAutodiffBackend, 1>::from_floats([4., 4.], &device).require_grad();
    let out = (lhs.clone() * 2.).hypot(rhs.clone() * 2.) + 1.;
    out.to_data()
        .assert_approx_eq::<f32>(&TensorData::from([11., 11.]), Tolerance::default());
    let grads = out.sum().backward();
    close(lhs.grad(&grads).unwrap(), [1.2, -1.2]);
    close(rhs.grad(&grads).unwrap(), [1.6, 1.6]);
}

#[cfg(feature = "ruda-tensor-router")]
#[test]
fn routed_forward_backward() {
    type Routed = ruda_tensor_router::Router<(TestBackend, TestBackend)>;
    type Ad = ruda_autodiff::Autodiff<Routed>;
    let device = Default::default();
    let lhs = Tensor::<Ad, 1>::from_floats([3., 5.], &device).require_grad();
    let rhs = Tensor::<Ad, 1>::from_floats([4., 12.], &device).require_grad();
    let out = lhs.clone().hypot(rhs.clone());
    out.to_data()
        .assert_approx_eq::<f32>(&TensorData::from([5., 13.]), Tolerance::default());
    let grads = out.sum().backward();
    lhs.grad(&grads)
        .unwrap()
        .into_data()
        .assert_approx_eq::<f32>(&TensorData::from([0.6, 5. / 13.]), Tolerance::default());
    rhs.grad(&grads)
        .unwrap()
        .into_data()
        .assert_approx_eq::<f32>(&TensorData::from([0.8, 12. / 13.]), Tolerance::default());
}
