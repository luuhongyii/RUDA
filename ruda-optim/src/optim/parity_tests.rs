use super::*;
use crate::{TestAutodiffBackend, TestBackend, grad_clipping::GradientClippingConfig};
use ruda_model::{
    module::Param,
    record::{BinBytesRecorder, FullPrecisionSettings, Record, Recorder},
    tensor::{DType, Tensor, TensorData, Tolerance, backend::Backend},
};
use ruda_nn::Linear;

fn assert_close<const D: usize>(
    actual: Tensor<TestBackend, D>,
    expected: impl Into<TensorData>,
    tolerance: f64,
) {
    actual
        .into_data()
        .assert_approx_eq::<f32>(&expected.into(), Tolerance::absolute(tolerance));
}

fn round_trip<B: Backend, R: Record<B>>(record: R, device: &B::Device) -> R {
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    let bytes =
        <BinBytesRecorder<FullPrecisionSettings> as Recorder<B>>::record(&recorder, record, ())
            .unwrap();
    <BinBytesRecorder<FullPrecisionSettings> as Recorder<B>>::load(&recorder, bytes, device)
        .unwrap()
}

#[test]
fn lion_two_steps_with_weight_decay() {
    let device = Default::default();
    let optimizer = LionConfig::new().with_weight_decay(0.1).build();
    let (tensor, state) = optimizer.step(
        0.1,
        Tensor::<TestBackend, 1>::from_floats([1., -2., 3.], &device),
        Tensor::from_floats([0.5, -0.25, 0.], &device),
        None,
    );
    assert_close(tensor.clone(), [0.89, -1.88, 2.97], 1e-6);
    assert_close(
        state.as_ref().unwrap().momentum.clone(),
        [0.005, -0.0025, 0.],
        1e-7,
    );
    let (tensor, state) = optimizer.step(
        0.1,
        tensor,
        Tensor::from_floats([-0.1, 0.5, -2.], &device),
        state,
    );
    assert_close(tensor, [0.9811, -1.9612, 3.0403], 1e-6);
    assert_close(state.unwrap().momentum, [0.00395, 0.002525, -0.02], 1e-7);
}

#[test]
fn lion_zero_gradient_preserves_zero_momentum() {
    let device = Default::default();
    let optimizer = LionConfig::new().build();
    let mut tensor = Tensor::<TestBackend, 1>::from_floats([1., -2.], &device);
    let mut state = None;
    for _ in 0..2 {
        (tensor, state) = optimizer.step(0.1, tensor, Tensor::zeros([2], &device), state);
    }
    assert_close(tensor, [1., -2.], 0.);
    assert_close(state.unwrap().momentum, [0., 0.], 0.);
}

#[test]
fn lamb_two_steps_with_trust_ratio_and_decay() {
    let device = Default::default();
    let optimizer = LambConfig::new().with_weight_decay(0.1).build();
    let (tensor, state) = optimizer.step(
        0.01,
        Tensor::<TestBackend, 1>::from_floats([1., -2., 3.], &device),
        Tensor::from_floats([0.1, -0.2, 0.3], &device),
        None,
    );
    assert_close(tensor.clone(), [0.9802435, -1.9784473, 2.9766512], 1e-6);
    let (tensor, state) = optimizer.step(
        0.01,
        tensor,
        Tensor::from_floats([-0.4, 0.5, -0.6], &device),
        state,
    );
    assert_close(tensor, [1.0127187, -1.9956441, 2.9814672], 1e-6);
    let state = state.unwrap();
    assert_eq!(state.time, 2);
    assert_close(state.moment_1, [-0.031, 0.032, -0.033], 1e-7);
    assert_close(state.moment_2, [0.00016999, 0.00028996, 0.00044991], 1e-8);
}

#[test]
fn lamb_without_trust_ratio() {
    let device = Default::default();
    let (tensor, _) = LambConfig::new()
        .with_weight_decay(0.1)
        .with_use_trust_ratio(false)
        .build()
        .step(
            0.01,
            Tensor::<TestBackend, 1>::from_floats([1., -2., 3.], &device),
            Tensor::from_floats([0.1, -0.2, 0.3], &device),
            None,
        );
    assert_close(tensor, [0.9890001, -1.988, 2.987], 1e-6);
}

#[test]
fn lamb_zero_norms_use_unit_trust_ratio() {
    let device = Default::default();
    let optimizer = LambConfig::new().build();
    let (zero, _) = optimizer.step(
        0.1,
        Tensor::<TestBackend, 2>::zeros([2, 2], &device),
        Tensor::zeros([2, 2], &device),
        None,
    );
    assert_close(zero, [[0., 0.], [0., 0.]], 0.);
    let (tensor, _) = optimizer.step(
        0.1,
        Tensor::<TestBackend, 2>::zeros([2, 2], &device),
        Tensor::ones([2, 2], &device),
        None,
    );
    assert_close(
        tensor,
        [[-0.0999999, -0.0999999], [-0.0999999, -0.0999999]],
        1e-6,
    );
    let (tensor, _) = optimizer.step(
        0.1,
        Tensor::<TestBackend, 1>::from_floats([1., -2.], &device),
        Tensor::zeros([2], &device),
        None,
    );
    assert_close(tensor, [1., -2.], 0.);
}

fn absolute_adafactor() -> Adafactor {
    AdafactorConfig::new()
        .with_epsilon_1(0.01)
        .with_clip_threshold(0.75)
        .with_relative_step(false)
        .with_scale_parameter(false)
        .build()
}

#[test]
fn adafactor_vector_two_steps() {
    let device = Default::default();
    let optimizer = absolute_adafactor();
    let (tensor, state) = optimizer.step(
        0.2,
        Tensor::<TestBackend, 1>::from_floats([1., -2., 3.], &device),
        Tensor::from_floats([0.2, -0.4, 0.7], &device),
        None,
    );
    assert_close(tensor.clone(), [0.8591300, -1.8472050, 2.8440855], 1e-6);
    let (tensor, state) = optimizer.step(
        0.2,
        tensor,
        Tensor::from_floats([-0.1, 0.8, -0.5], &device),
        state,
    );
    assert_close(tensor, [0.9511715, -2.0468662, 2.982519], 1e-6);
    let state = state.unwrap();
    assert_eq!(state.time, 2);
    assert!(state.column_second_moment.is_none());
    assert_close(
        state.second_moment,
        [0.032769524, 0.4456876, 0.3621562],
        1e-7,
    );
}

#[test]
fn adafactor_matrix_two_steps_and_factored_state() {
    let device = Default::default();
    let optimizer = absolute_adafactor();
    let (tensor, state) = optimizer.step(
        0.2,
        Tensor::<TestBackend, 2>::from_floats([[1., -2., 3.], [-4., 5., -6.]], &device),
        Tensor::from_floats([[0.2, -0.4, 0.6], [0.8, -1., 1.2]], &device),
        None,
    );
    assert_close(
        tensor.clone(),
        [
            [0.90609723, -1.8553506, 2.8252919],
            [-4.1636004, 5.157508, -6.152191],
        ],
        1e-6,
    );
    let (tensor, state) = optimizer.step(
        0.2,
        tensor,
        Tensor::from_floats([[-0.5, 0.25, 1.], [0.125, -0.75, 0.375]], &device),
        state,
    );
    assert_close(
        tensor,
        [
            [1.109898, -1.929492, 2.5945199],
            [-4.202498, 5.3273153, -6.218259],
        ],
        1e-6,
    );
    let state = state.unwrap();
    assert_eq!(state.time, 2);
    assert_eq!(state.second_moment.dims(), [2, 1]);
    assert_close(state.second_moment, [[0.34073257], [0.584606]], 1e-7);
    let column = state.column_second_moment.unwrap();
    assert_eq!(column.dims(), [1, 3]);
    assert_close(column, [[0.23100203, 0.43636158, 0.72064424]], 1e-7);
}

#[test]
fn adafactor_higher_rank_preserves_independent_factors() {
    let device = Default::default();
    let optimizer = AdafactorConfig::new()
        .with_epsilon_1(0.01)
        .with_clip_threshold(10.)
        .with_relative_step(false)
        .with_scale_parameter(false)
        .build();
    let (tensor, state) = optimizer.step(
        0.2,
        Tensor::<TestBackend, 3>::zeros([2, 2, 3], &device),
        Tensor::from_floats(
            [
                [[1., 2., 3.], [4., 5., 6.]],
                [[2., -1., 4.], [0.5, 3., -2.]],
            ],
            &device,
        ),
        None,
    );
    assert_close(
        tensor,
        [
            [
                [-0.12350519, -0.1891674, -0.22781561],
                [-0.21083567, -0.20182975, -0.19445199],
            ],
            [
                [-0.24725027, 0.080702476, -0.22837506],
                [-0.077785276, -0.30466932, 0.1436942],
            ],
        ],
        1e-6,
    );
    let state = state.unwrap();
    assert_eq!(state.second_moment.dims(), [2, 2, 1]);
    assert_close(
        state.second_moment,
        [[[4.6766667], [25.676666]], [[7.01], [4.4266667]]],
        3e-6,
    );
    let column = state.column_second_moment.unwrap();
    assert_eq!(column.dims(), [2, 1, 3]);
    assert_close(
        column,
        [[[8.51, 14.51, 22.51]], [[2.135, 5.01, 10.01]]],
        3e-6,
    );
}

#[test]
fn adafactor_relative_and_absolute_learning_rates() {
    let device = Default::default();
    let optimizer = AdafactorConfig::new().with_scale_parameter(false).build();
    let tensor = Tensor::<TestBackend, 1>::from_floats([2., -2.], &device);
    let grad = Tensor::ones([2], &device);
    let (small, _) = optimizer.step(0.1, tensor.clone(), grad.clone(), None);
    assert_close(small, [1.9, -2.1], 1e-6);
    let (first, state) = optimizer.step(2., tensor.clone(), grad.clone(), None);
    assert_close(first.clone(), [1., -3.], 1e-6);
    let (second, _) = optimizer.step(2., first, grad.clone(), state);
    assert_close(second, [0.29289323, -3.7071068], 1e-6);
    let (absolute, _) = AdafactorConfig::new()
        .with_relative_step(false)
        .with_scale_parameter(false)
        .build()
        .step(2., tensor, grad, None);
    assert_close(absolute, [0., -4.], 1e-6);
}

#[test]
fn adafactor_parameter_scale_and_epsilon_floor() {
    let device = Default::default();
    let optimizer = AdafactorConfig::new().with_epsilon_2(0.1).build();
    let (tensor, _) = optimizer.step(
        0.01,
        Tensor::<TestBackend, 1>::from_floats([3., 4.], &device),
        Tensor::ones([2], &device),
        None,
    );
    assert_close(tensor, [2.9646447, 3.9646447], 1e-6);
    let (tensor, _) = optimizer.step(
        0.2,
        Tensor::<TestBackend, 1>::zeros([2], &device),
        Tensor::ones([2], &device),
        None,
    );
    assert_close(tensor, [-0.02, -0.02], 1e-7);
}

#[test]
fn adafactor_update_clipping_and_decay() {
    let device = Default::default();
    let (tensor, _) = AdafactorConfig::new()
        .with_clip_threshold(0.25)
        .with_scale_parameter(false)
        .build()
        .step(
            0.2,
            Tensor::<TestBackend, 1>::from_floats([1., -2.], &device),
            Tensor::from_floats([1., -1.], &device),
            None,
        );
    assert_close(tensor, [0.95, -1.95], 1e-6);
    let (tensor, _) = AdafactorConfig::new().with_weight_decay(0.1).build().step(
        2.,
        Tensor::<TestBackend, 1>::from_floats([3., 4.], &device),
        Tensor::ones([2], &device),
        None,
    );
    assert_close(tensor, [-1.1355339, -0.33553392], 1e-6);
}

#[test]
fn adafactor_zero_gradients_remain_finite() {
    let device = Default::default();
    let optimizer = AdafactorConfig::new().build();
    let mut vector = Tensor::<TestBackend, 1>::zeros([3], &device);
    let mut matrix = Tensor::<TestBackend, 2>::zeros([2, 3], &device);
    let (mut vector_state, mut matrix_state) = (None, None);
    for _ in 0..2 {
        (vector, vector_state) =
            optimizer.step(0.1, vector, Tensor::zeros([3], &device), vector_state);
        (matrix, matrix_state) =
            optimizer.step(0.1, matrix, Tensor::zeros([2, 3], &device), matrix_state);
    }
    assert_close(vector, [0., 0., 0.], 0.);
    assert_close(matrix, [[0., 0., 0.], [0., 0., 0.]], 0.);
}

#[test]
fn adafactor_half_parameters_keep_fp32_state_through_record() {
    let device = Default::default();
    let optimizer = AdafactorConfig::new().with_scale_parameter(false).build();
    for dtype in [DType::F16, DType::BF16] {
        let mut tensor =
            Tensor::<TestBackend, 2>::from_floats([[1., -2.], [3., -4.]], &device).cast(dtype);
        let mut state = None;
        for time in 1..=2 {
            (tensor, state) = optimizer.step(
                0.1,
                tensor,
                Tensor::<TestBackend, 2>::zeros([2, 2], &device).cast(dtype),
                state,
            );
            assert_eq!(tensor.dtype(), dtype);
            assert_close(tensor.clone(), [[1., -2.], [3., -4.]], 0.);
            let saved = state.take().unwrap();
            let moved = <Adafactor as SimpleOptimizer<TestBackend>>::to_device(saved, &device);
            let restored: AdafactorState<TestBackend, 2> = round_trip(moved, &device);
            assert_eq!(restored.time, time);
            assert_eq!(restored.second_moment.dtype(), DType::F32);
            assert_close(restored.second_moment.clone(), [[1e-30], [1e-30]], 1e-36);
            let column = restored.column_second_moment.as_ref().unwrap();
            assert_eq!(column.dtype(), DType::F32);
            assert_close(column.clone(), [[1e-30, 1e-30]], 1e-36);
            state = Some(restored);
        }
    }
}

#[test]
fn lion_and_lamb_records_preserve_half_dtype() {
    let device = Default::default();
    for dtype in [DType::F16, DType::BF16] {
        let momentum = Tensor::<TestBackend, 1>::from_floats([0.25, -0.5], &device).cast(dtype);
        let lion = LionState::new(momentum.clone());
        let lion = <Lion as SimpleOptimizer<TestBackend>>::to_device(lion, &device);
        let lion: LionState<TestBackend, 1> = round_trip(lion, &device);
        assert_eq!(lion.momentum.dtype(), dtype);
        assert_close(lion.momentum, [0.25, -0.5], 0.);
        let lamb = LambState {
            time: 7,
            moment_1: momentum.clone(),
            moment_2: momentum.square(),
        };
        let lamb = <Lamb as SimpleOptimizer<TestBackend>>::to_device(lamb, &device);
        let lamb: LambState<TestBackend, 1> = round_trip(lamb, &device);
        assert_eq!(lamb.time, 7);
        assert_eq!(lamb.moment_1.dtype(), dtype);
        assert_eq!(lamb.moment_2.dtype(), dtype);
        assert_close(lamb.moment_1, [0.25, -0.5], 0.);
        assert_close(lamb.moment_2, [0.0625, 0.25], 0.);
    }
}

fn linear() -> Linear<TestAutodiffBackend> {
    let device = Default::default();
    Linear {
        weight: Param::from_data(TensorData::from([[1.0f32, -2.], [3., -4.]]), &device),
        bias: Some(Param::from_data(TensorData::from([0.5f32, -0.5]), &device)),
    }
}

fn module_step<O: Optimizer<Linear<TestAutodiffBackend>, TestAutodiffBackend>>(
    optimizer: &mut O,
    model: Linear<TestAutodiffBackend>,
    input: [[f32; 2]; 1],
) -> Linear<TestAutodiffBackend> {
    let output = model.forward(Tensor::<TestAutodiffBackend, 2>::from_floats(
        input,
        &Default::default(),
    ));
    let gradients = GradientsParams::from_grads(output.sum().backward(), &model);
    optimizer.step(0.1, model, gradients)
}

fn assert_module_resume<O: Optimizer<Linear<TestAutodiffBackend>, TestAutodiffBackend>>(
    initial: O,
) {
    let device = Default::default();
    let mut uninterrupted = initial.clone();
    let model = module_step(&mut uninterrupted, linear(), [[0.25, -0.75]]);
    let mut restored = initial.load_record(round_trip::<TestAutodiffBackend, _>(
        uninterrupted.to_record(),
        &device,
    ));
    let mut expected = model.clone();
    let mut actual = model;
    for input in [[[-1., 0.5]], [[0.125, -2.]]] {
        expected = module_step(&mut uninterrupted, expected, input);
        actual = module_step(&mut restored, actual, input);
        assert_eq!(expected.weight.id, actual.weight.id);
        actual
            .weight
            .val()
            .to_data()
            .assert_approx_eq::<f32>(&expected.weight.val().to_data(), Tolerance::absolute(1e-6));
        actual
            .bias
            .as_ref()
            .unwrap()
            .val()
            .to_data()
            .assert_approx_eq::<f32>(
                &expected.bias.as_ref().unwrap().val().to_data(),
                Tolerance::absolute(1e-6),
            );
    }
}

#[test]
fn lion_module_resume() {
    assert_module_resume(
        LionConfig::new()
            .with_weight_decay(0.1)
            .with_grad_clipping(Some(GradientClippingConfig::Value(0.1)))
            .init(),
    );
}

#[test]
fn lamb_module_resume() {
    assert_module_resume(
        LambConfig::new()
            .with_weight_decay(0.1)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(0.5)))
            .init(),
    );
}

#[test]
fn adafactor_module_resume() {
    assert_module_resume(
        AdafactorConfig::new()
            .with_weight_decay(0.1)
            .with_grad_clipping(Some(GradientClippingConfig::Value(0.1)))
            .init(),
    );
}

#[test]
fn module_gradient_clipping_matches_explicit_clipped_updates() {
    let device = Default::default();
    macro_rules! check {
        ($config:expr) => {{
            let config = $config.with_grad_clipping(Some(GradientClippingConfig::Value(0.1)));
            let original = linear();
            let bare = config.build();
            let (weight, _) = bare.step(
                0.1,
                original.weight.val().inner(),
                Tensor::from_floats([[0.1, 0.1], [-0.1, -0.1]], &device),
                None,
            );
            let (bias, _) = bare.step(
                0.1,
                original.bias.as_ref().unwrap().val().inner(),
                Tensor::from_floats([0.1, 0.1], &device),
                None,
            );
            let mut optimizer = config.init();
            assert!(optimizer.has_gradient_clipping());
            let actual = module_step(&mut optimizer, original, [[0.25, -0.75]]);
            assert_close(actual.weight.val().inner(), weight.into_data(), 1e-6);
            assert_close(actual.bias.unwrap().val().inner(), bias.into_data(), 1e-6);
        }};
    }
    check!(LionConfig::new());
    check!(LambConfig::new());
    check!(
        AdafactorConfig::new()
            .with_epsilon_1(1.0)
            .with_relative_step(false)
            .with_scale_parameter(false)
    );
}

#[test]
#[should_panic(expected = "epsilon_1 must be positive and finite")]
fn adafactor_rejects_zero_epsilon() {
    AdafactorConfig::new().with_epsilon_1(0.).build();
}

#[test]
#[should_panic(expected = "epsilon_2 must be positive and finite")]
fn adafactor_rejects_negative_scale_epsilon() {
    AdafactorConfig::new().with_epsilon_2(-1.).build();
}

#[test]
#[should_panic(expected = "clip_threshold must be positive and finite")]
fn adafactor_rejects_nan_clip() {
    AdafactorConfig::new().with_clip_threshold(f32::NAN).build();
}

#[test]
#[should_panic(expected = "decay_rate must be negative and finite")]
fn adafactor_rejects_nonnegative_decay() {
    AdafactorConfig::new().with_decay_rate(0.).build();
}

#[test]
#[should_panic(expected = "weight_decay must be nonnegative and finite")]
fn adafactor_rejects_infinite_weight_decay() {
    AdafactorConfig::new()
        .with_weight_decay(f32::INFINITY)
        .build();
}
