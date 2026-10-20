use crate::{TestAutodiffBackend, TestBackend};
use crate::loss::{CosineEmbeddingLossConfig, CrossEntropyLossConfig};
use alloc::vec;
use ruda_model::tensor::{DType, Int, Tensor, TensorData, Tolerance, linalg::cosine_similarity};

#[test]
fn cross_entropy_padding_values_and_gradients_all_modes() {
    let device = Default::default();
    for dtype in [DType::F32, DType::F64] {
        for logits in [true, false] {
            for smoothing in [None, Some(0.0), Some(0.1)] {
                for weighted in [false, true] {
                    let config = CrossEntropyLossConfig::new()
                        .with_pad_tokens(Some(vec![2, 2]))
                        .with_weights(weighted.then(|| vec![1.0, 2.0, 4.0]))
                        .with_smoothing(smoothing)
                        .with_logits(logits);
                    let valid = Tensor::<TestAutodiffBackend, 2>::from_floats(
                        [[0.7, 0.2, 0.1], [0.1, 0.7, 0.2]], &device,
                    ).cast(dtype).require_grad();
                    let padded = Tensor::<TestAutodiffBackend, 2>::from_floats(
                        [[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.3, 0.5], [0.3, 0.2, 0.5]],
                        &device,
                    ).cast(dtype).require_grad();
                    let expected = config.init(&device).forward(
                        valid.clone(), Tensor::<TestAutodiffBackend, 1, Int>::from_data([0, 1], &device),
                    );
                    let actual = config.init(&device).forward(
                        padded.clone(), Tensor::<TestAutodiffBackend, 1, Int>::from_data([0, 1, 2, 2], &device),
                    );
                    assert_eq!(actual.dtype(), dtype);
                    actual.to_data().assert_approx_eq::<f64>(
                        &expected.to_data(), Tolerance::rel_abs(2e-5, 2e-6),
                    );
                    let expected_grad = valid.grad(&expected.backward()).unwrap();
                    let actual_grad = padded.grad(&actual.backward()).unwrap();
                    actual_grad.clone().slice([0..2, 0..3]).to_data().assert_approx_eq::<f64>(
                        &expected_grad.to_data(), Tolerance::rel_abs(2e-5, 2e-6),
                    );
                    actual_grad.slice([2..4, 0..3]).to_data().convert::<f64>()
                        .assert_eq(&TensorData::from([[0.0f64; 3]; 2]), true);
                }
            }
        }
    }
}

#[test]
fn cross_entropy_multiple_empty_and_all_padding() {
    let device = Default::default();
    let input = Tensor::<TestBackend, 2>::from_floats(
        [[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]], &device,
    );
    let targets = Tensor::<TestBackend, 1, Int>::from_data([0, 1, 2], &device);
    for logits in [true, false] {
        for smoothing in [None, Some(0.0), Some(0.1)] {
            for weighted in [false, true] {
                let config = CrossEntropyLossConfig::new().with_logits(logits)
                    .with_smoothing(smoothing)
                    .with_weights(weighted.then(|| vec![1.0, 2.0, 4.0]));
                let expected = config.init(&device).forward(
                    input.clone().slice([0..1, 0..3]), Tensor::from_data([0], &device),
                );
                let actual = config.clone().with_pad_tokens(Some(vec![1, 2])).init(&device)
                    .forward(input.clone(), targets.clone());
                actual.to_data().assert_approx_eq::<f32>(&expected.to_data(), Tolerance::default());
                let empty = config.clone().with_pad_tokens(Some(vec![])).init(&device)
                    .forward(input.clone(), targets.clone());
                let unmasked = config.init(&device).forward(input.clone(), targets.clone());
                empty.to_data().assert_approx_eq::<f32>(&unmasked.to_data(), Tolerance::default());
                let all = config.with_pad_tokens(Some(vec![0, 1, 2])).init(&device)
                    .forward(input.clone(), targets.clone()).into_data();
                assert!(all.as_slice::<f32>().unwrap()[0].is_nan());
            }
        }
    }
}

#[test]
fn cross_entropy_smoothed_zero_probabilities() {
    let device = Default::default();
    let input = Tensor::<TestBackend, 2>::from_floats([
        [0.1, 0.2, 0.7, 0.0, 0.0], [0.7, 0.1, 0.1, 0.1, 0.0],
        [0.2, 0.2, 0.2, 0.2, 0.2], [0.0, 0.3, 0.3, 0.2, 0.2],
    ], &device);
    let targets = Tensor::<TestBackend, 1, Int>::from_data([2, 0, 4, 1], &device);
    let config = CrossEntropyLossConfig::new().with_logits(false);
    let ordinary = config.init(&device).forward(input.clone(), targets.clone());
    let zero = config.clone().with_smoothing(Some(0.0)).init(&device)
        .forward(input.clone(), targets.clone());
    zero.to_data().assert_approx_eq::<f32>(&ordinary.to_data(), Tolerance::default());
    let smoothed = config.with_smoothing(Some(0.1)).init(&device).forward(input, targets);
    smoothed.to_data().assert_approx_eq::<f32>(&TensorData::from([1.7929223]), Tolerance::default());
}

#[test]
fn cross_entropy_fixed_gradient_oracle() {
    let device = Default::default();
    let logits = Tensor::<TestAutodiffBackend, 2>::from_floats(
        [[2.0, 0.0, -1.0], [0.0, 0.0, 0.0]], &device,
    ).require_grad();
    let output = CrossEntropyLossConfig::new().with_pad_tokens(Some(vec![2])).init(&device)
        .forward(logits.clone(), Tensor::from_data([0, 2], &device));
    logits.grad(&output.backward()).unwrap().to_data().assert_approx_eq::<f32>(
        &TensorData::from([[-0.1562053, 0.1141952, 0.04201007], [0.0, 0.0, 0.0]]),
        Tolerance::relative(1e-4),
    );
}

#[test]
fn cosine_zero_vectors_keep_dtype_and_embedding_values() {
    let device = Default::default();
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let input = Tensor::<TestBackend, 2>::zeros([2, 3], (&device, dtype));
        let output = cosine_similarity(input.clone(), input.clone(), -1, None);
        assert_eq!(output.dtype(), dtype);
        output.to_data().convert::<f32>().assert_eq(&TensorData::from([[0.0f32], [0.0]]), true);
        let loss = CosineEmbeddingLossConfig::new().init().forward_no_reduction(
            input.clone(), input, Tensor::<TestBackend, 1, Int>::from_data([1, -1], &device),
        );
        loss.to_data().convert::<f32>().assert_eq(&TensorData::from([1.0f32, 0.0]), true);
    }
}

#[test]
fn cosine_broadcast_and_backward() {
    let device = Default::default();
    let first = Tensor::<TestAutodiffBackend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device).require_grad();
    let second = Tensor::<TestAutodiffBackend, 2>::from_floats([[3.0, 4.0]], &device).require_grad();
    let output = cosine_similarity(first.clone(), second.clone(), -1, None);
    output.to_data().assert_approx_eq::<f32>(
        &TensorData::from([[0.9838699], [1.0]]), Tolerance::rel_abs(2e-5, 2e-6),
    );
    let gradients = output.sum().backward();
    first.grad(&gradients).unwrap().to_data().assert_approx_eq::<f32>(
        &TensorData::from([[0.07155418, -0.03577709], [0.0, 0.0]]), Tolerance::rel_abs(2e-5, 2e-6),
    );
    second.grad(&gradients).unwrap().to_data().assert_approx_eq::<f32>(
        &TensorData::from([[-0.02862167, 0.02146625]]), Tolerance::rel_abs(2e-5, 2e-6),
    );
}

#[test]
#[should_panic(expected = "Cosine Similarity dimension is out of bounds")]
fn cosine_rejects_out_of_range_negative_axis() {
    let device = Default::default();
    let input = Tensor::<TestBackend, 2>::ones([2, 3], &device);
    let _ = cosine_similarity(input.clone(), input, -3, None);
}
