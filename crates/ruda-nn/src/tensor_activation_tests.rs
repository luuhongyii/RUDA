use crate::{Identity, TestAutodiffBackend, TestBackend};
use crate::activation::{ActivationConfig, HardtanhConfig, ThresholdConfig};
use ruda_model::tensor::{Bool, Int, Tensor, TensorData, Tolerance};

fn check_activation(config: ActivationConfig, reference: fn(f64) -> (f64, f64)) {
    let device = Default::default();
    let values = [-4.0f64, -0.4, 0.3, 2.0, 8.0];
    let input = Tensor::<TestAutodiffBackend, 1>::from_floats(values.map(|x| x as f32), &device)
        .require_grad();
    let output = config.init(&device).forward(input.clone());
    output.to_data().assert_approx_eq::<f32>(
        &TensorData::from(values.map(|x| reference(x).0 as f32)),
        Tolerance::rel_abs(2e-5, 2e-6),
    );
    let gradients = output.sum().backward();
    input.grad(&gradients).unwrap().to_data().assert_approx_eq::<f32>(
        &TensorData::from(values.map(|x| reference(x).1 as f32)),
        Tolerance::rel_abs(2e-5, 2e-6),
    );
}

#[test]
fn relu6_forward_backward() {
    check_activation(ActivationConfig::Relu6, |x| {
        (x.clamp(0.0, 6.0), if x > 0.0 && x < 6.0 { 1.0 } else { 0.0 })
    });
}

#[test]
fn hardtanh_forward_backward() {
    check_activation(HardtanhConfig::new().into(), |x| {
        (x.clamp(-1.0, 1.0), if x > -1.0 && x < 1.0 { 1.0 } else { 0.0 })
    });
}

#[test]
fn log_sigmoid_forward_backward() {
    check_activation(ActivationConfig::LogSigmoid, |x| {
        (-(-x).exp().ln_1p(), 1.0 / (1.0 + x.exp()))
    });
}

#[test]
fn mish_forward_backward() {
    check_activation(ActivationConfig::Mish, |x| {
        let t = x.exp().ln_1p().tanh();
        (x * t, t + x * (1.0 - t * t) / (1.0 + (-x).exp()))
    });
}

#[test]
fn tanhshrink_forward_backward() {
    check_activation(ActivationConfig::Tanhshrink, |x| (x - x.tanh(), x.tanh().powi(2)));
}

#[test]
fn threshold_forward_backward() {
    check_activation(ThresholdConfig::new(0.5, -2.0).into(), |x| {
        if x > 0.5 { (x, 1.0) } else { (-2.0, 0.0) }
    });
}

#[test]
fn threshold_includes_boundary() {
    let device = Default::default();
    let x = Tensor::<TestAutodiffBackend, 1>::from_floats([-2.0, 1.0, 2.0], &device).require_grad();
    let y = ThresholdConfig::new(1.0, -3.0).init().forward(x.clone());
    y.to_data().assert_eq(&TensorData::from([-3.0f32, -3.0, 2.0]), true);
    x.grad(&y.sum().backward()).unwrap().to_data()
        .assert_eq(&TensorData::from([0.0f32, 0.0, 1.0]), true);
}

#[test]
fn identity_preserves_values_and_gradients() {
    let device = Default::default();
    let x = Tensor::<TestAutodiffBackend, 2>::from_floats([[1.0, -2.0], [3.0, 4.0]], &device)
        .require_grad();
    let y = Identity::new().forward(x.clone());
    y.to_data().assert_eq(&x.to_data(), true);
    x.grad(&y.sum().backward()).unwrap().to_data()
        .assert_eq(&TensorData::from([[1.0f32, 1.0], [1.0, 1.0]]), true);
}

#[test]
fn mask_select_all_kinds_and_strided_layout() {
    let device = Default::default();
    let mask = Tensor::<TestBackend, 2, Bool>::from_data([[true, false], [false, true], [true, false]], &device);
    let x = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device).transpose();
    x.mask_select(mask.clone()).to_data().assert_eq(&TensorData::from([1.0f32, 5.0, 3.0]), true);
    let x = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3], [4, 5, 6]], &device).transpose();
    x.mask_select(mask.clone()).to_data().assert_eq(&TensorData::from([1, 5, 3]), false);
    let x = Tensor::<TestBackend, 2, Bool>::from_data([[false, true], [true, true], [false, true]], &device);
    x.mask_select(mask).to_data().assert_eq(&TensorData::from([false, true, false]), false);
}

#[test]
fn mask_select_empty_and_all() {
    let device = Default::default();
    let x = Tensor::<TestBackend, 1, Int>::from_ints([3, 7, 8], &device);
    let empty = x.clone().mask_select(Tensor::from_data([false, false, false], &device));
    assert_eq!(empty.dims(), [0]);
    assert_eq!(empty.to_data().num_elements(), 0);
    x.clone().mask_select(Tensor::from_data([true, true, true], &device)).to_data()
        .assert_eq(&x.to_data(), true);
    let x = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device).transpose();
    let empty = x.mask_select(Tensor::from_data([[false, false], [false, false]], &device));
    assert_eq!(empty.dims(), [0]);
    assert_eq!(empty.to_data().num_elements(), 0);
    let x = Tensor::<TestBackend, 1, Bool>::from_data([true, false], &device);
    let empty = x.mask_select(Tensor::from_data([false, false], &device));
    assert_eq!(empty.dims(), [0]);
    assert_eq!(empty.to_data().num_elements(), 0);
}

#[test]
fn mask_select_backward() {
    let device = Default::default();
    let x = Tensor::<TestAutodiffBackend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device).require_grad();
    let mask = Tensor::from_data([[false, true], [true, false]], &device);
    let gradients = x.clone().mask_select(mask).sum().backward();
    x.grad(&gradients).unwrap().to_data()
        .assert_eq(&TensorData::from([[0.0f32, 1.0], [1.0, 0.0]]), true);
    let mask = Tensor::from_data([[false, false], [false, false]], &device);
    let sum = x.clone().mask_select(mask).sum();
    sum.to_data().assert_eq(&TensorData::from([0.0f32]), true);
    x.grad(&sum.backward()).unwrap().to_data()
        .assert_eq(&TensorData::from([[0.0f32, 0.0], [0.0, 0.0]]), true);
}

#[test]
#[should_panic(expected = "mask_select requires equal tensor and mask shapes")]
fn mask_select_rejects_different_shapes() {
    let device = Default::default();
    Tensor::<TestBackend, 1>::from_floats([1.0, 2.0], &device)
        .mask_select(Tensor::from_data([true], &device));
}

#[test]
fn integer_square_preserves_integer_precision() {
    let device = Default::default();
    Tensor::<TestBackend, 1, Int>::from_ints([-16, 0, 4001, 46340], &device).square().to_data()
        .assert_eq(&TensorData::from([256, 0, 16008001, 2147395600]), false);
}
