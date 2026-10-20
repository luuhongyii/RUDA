use crate::{
    CosineSimilarityConfig, Fold4dConfig, PairwiseDistanceConfig, PixelShuffleConfig,
    PixelUnshuffleConfig, TestAutodiffBackend, TestBackend,
    loss::{
        GaussianNLLLossConfig, HingeEmbeddingLossConfig, MarginRankingLossConfig,
        MultiMarginLossConfig, Reduction, SoftMarginLoss, TripletMarginLossConfig,
    },
};
use alloc::vec;
use alloc::vec::Vec;
use ruda_model::tensor::{
    DType, Int, Tensor, TensorData, Tolerance, activation::softplus_with_threshold,
};

fn close<const D: usize>(actual: Tensor<TestBackend, D>, expected: impl Into<TensorData>) {
    actual
        .into_data()
        .assert_approx_eq::<f32>(&expected.into(), Tolerance::rel_abs(2e-5, 2e-6));
}

#[test]
fn gaussian_broadcast_variance_backward() {
    let device = Default::default();
    let input =
        Tensor::<TestAutodiffBackend, 2>::from_floats([[1., 2.], [3., 4.]], &device).require_grad();
    let target = Tensor::<TestAutodiffBackend, 2>::zeros([2, 2], &device).require_grad();
    let var = Tensor::<TestAutodiffBackend, 2>::from_floats([[0.5], [2.]], &device).require_grad();
    let output = GaussianNLLLossConfig::new().init().forward(
        input.clone(),
        target.clone(),
        var.clone(),
        Reduction::Sum,
    );
    let grads = output.backward();
    close(input.grad(&grads).unwrap(), [[2., 4.], [1.5, 2.]]);
    close(target.grad(&grads).unwrap(), [[-2., -4.], [-1.5, -2.]]);
    close(var.grad(&grads).unwrap(), [[-8.], [-2.625]]);
}

#[test]
fn gaussian_variance_clamp_backward() {
    let device = Default::default();
    let input = Tensor::<TestAutodiffBackend, 1>::from_floats([1.], &device).require_grad();
    let var = Tensor::<TestAutodiffBackend, 1>::from_floats([0.25], &device).require_grad();
    let output = GaussianNLLLossConfig::new().with_eps(0.5).init().forward(
        input.clone(),
        Tensor::zeros([1], &device),
        var.clone(),
        Reduction::Sum,
    );
    let grads = output.backward();
    close(input.grad(&grads).unwrap(), [2.]);
    close(var.grad(&grads).unwrap(), [0.]);
}

#[test]
fn hinge_embedding_backward() {
    let device = Default::default();
    let input =
        Tensor::<TestAutodiffBackend, 1>::from_floats([0.5, 1.5, 3.], &device).require_grad();
    let target = Tensor::from_floats([1., -1., -1.], &device);
    let output = HingeEmbeddingLossConfig::new()
        .with_margin(2.)
        .init()
        .forward(input.clone(), target, Reduction::Sum);
    close(input.grad(&output.backward()).unwrap(), [1., -1., 0.]);
}

#[test]
fn margin_ranking_backward_all_operands() {
    let device = Default::default();
    let first = Tensor::<TestAutodiffBackend, 1>::from_floats([1., 3.], &device).require_grad();
    let second = Tensor::<TestAutodiffBackend, 1>::from_floats([2., 1.], &device).require_grad();
    let target = Tensor::<TestAutodiffBackend, 1>::ones([2], &device).require_grad();
    let output = MarginRankingLossConfig::new()
        .with_margin(0.5)
        .init()
        .forward(
            first.clone(),
            second.clone(),
            target.clone(),
            Reduction::Sum,
        );
    let grads = output.backward();
    close(first.grad(&grads).unwrap(), [-1., 0.]);
    close(second.grad(&grads).unwrap(), [1., 0.]);
    close(target.grad(&grads).unwrap(), [1., 0.]);
}

#[test]
fn multi_margin_backward_both_exponents() {
    let device = Default::default();
    for (p, expected) in [
        (1, [[1. / 3., 1. / 3., -2. / 3.]]),
        (2, [[0.8 / 3., 1. / 3., -0.6]]),
    ] {
        let input = Tensor::<TestAutodiffBackend, 2>::from_floats([[0.1, 0.2, 0.7]], &device)
            .require_grad();
        let target = Tensor::<TestAutodiffBackend, 1, Int>::from_ints([2], &device);
        let output = MultiMarginLossConfig::new().with_p(p).init().forward(
            input.clone(),
            target,
            Reduction::Sum,
        );
        close(input.grad(&output.backward()).unwrap(), expected);
    }
}

#[test]
fn triplet_margin_backward_all_operands() {
    let device = Default::default();
    let anchor = Tensor::<TestAutodiffBackend, 2>::zeros([1, 2], &device).require_grad();
    let positive =
        Tensor::<TestAutodiffBackend, 2>::from_floats([[3., 4.]], &device).require_grad();
    let negative =
        Tensor::<TestAutodiffBackend, 2>::from_floats([[0., 2.]], &device).require_grad();
    let output = TripletMarginLossConfig::new().init().forward(
        anchor.clone(),
        positive.clone(),
        negative.clone(),
        Reduction::Sum,
    );
    close(output.clone().inner(), [4.]);
    let grads = output.backward();
    close(anchor.grad(&grads).unwrap(), [[-0.6, 0.2]]);
    close(positive.grad(&grads).unwrap(), [[0.6, 0.8]]);
    close(negative.grad(&grads).unwrap(), [[0., -1.]]);
}

#[test]
fn soft_margin_saturation_and_negative_tail_backward() {
    let device = Default::default();
    let input = Tensor::<TestAutodiffBackend, 1>::from_floats([-100., -20., -5., 20.], &device)
        .require_grad();
    let output =
        SoftMarginLoss::new().forward_no_reduction(input.clone(), Tensor::ones([4], &device));
    output.to_data().assert_approx_eq::<f32>(
        &TensorData::from([100., 20., 5.00671535, 2.0611536e-9]),
        Tolerance::rel_abs(2e-5, 1e-10),
    );
    let grads = output.sum().backward();
    input
        .grad(&grads)
        .unwrap()
        .to_data()
        .assert_approx_eq::<f32>(
            &TensorData::from([-1., -1., -0.99330715, -2.0611536e-9]),
            Tolerance::rel_abs(2e-5, 1e-10),
        );
}

#[test]
fn softplus_dtype_exp_limits_and_backward() {
    let device = Default::default();
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let input =
            Tensor::<TestAutodiffBackend, 1>::from_floats([-5., 0., 5., 13., 100.], &device)
                .cast(dtype)
                .require_grad();
        let output = softplus_with_threshold(input.clone(), 1., 1000.);
        assert_eq!(output.dtype(), dtype);
        output.to_data().assert_approx_eq::<f32>(
            &TensorData::from([0.00671535, 0.69314718, 5.00671535, 13.0000023, 100.]),
            Tolerance::rel_abs(0.01, 0.01),
        );
        input
            .grad(&output.sum().backward())
            .unwrap()
            .to_data()
            .assert_approx_eq::<f32>(
                &TensorData::from([0.00669285, 0.5, 0.99330715, 0.99999774, 1.]),
                Tolerance::rel_abs(0.02, 0.01),
            );
    }
}

#[test]
fn pairwise_distance_general_norm_degrees() {
    let device = Default::default();
    for (p, expected) in [
        (-2., 2.4),
        (0., 2.),
        (0.5, (3.0f64.sqrt() + 2.).powi(2)),
        (1., 7.),
        (2., 5.),
        (4., 337.0f64.powf(0.25)),
        (f64::INFINITY, 4.),
        (f64::NEG_INFINITY, 3.),
    ] {
        let output = PairwiseDistanceConfig::new()
            .with_p(p)
            .with_eps(0.)
            .init()
            .forward(
                Tensor::<TestBackend, 2>::from_floats([[3., -4.]], &device),
                Tensor::zeros([1, 2], &device),
            );
        close(output, [expected as f32]);
    }
}

#[test]
fn pairwise_negative_even_norm_backward() {
    let device = Default::default();
    let first = Tensor::<TestAutodiffBackend, 2>::from_floats([[3., -4.]], &device).require_grad();
    let second = Tensor::<TestAutodiffBackend, 2>::zeros([1, 2], &device).require_grad();
    let output = PairwiseDistanceConfig::new()
        .with_p(-2.)
        .with_eps(0.)
        .init()
        .forward(first.clone(), second.clone());
    let grads = output.sum().backward();
    close(first.grad(&grads).unwrap(), [[0.512, -0.216]]);
    close(second.grad(&grads).unwrap(), [[-0.512, 0.216]]);
}

#[test]
fn cosine_broadcast_backward_and_tiny_epsilon() {
    let device = Default::default();
    let first = Tensor::<TestAutodiffBackend, 2>::from_floats([[1., 0.]], &device).require_grad();
    let second =
        Tensor::<TestAutodiffBackend, 2>::from_floats([[0., 1.], [1., 0.]], &device).require_grad();
    let output = CosineSimilarityConfig::new()
        .init()
        .forward(first.clone(), second.clone());
    close(output.clone().inner(), [0., 1.]);
    let grads = output.sum().backward();
    close(first.grad(&grads).unwrap(), [[0., 1.]]);
    close(second.grad(&grads).unwrap(), [[1., 0.], [0., 0.]]);
    let zeros = Tensor::<TestBackend, 2>::zeros([2, 2], &device);
    close(
        CosineSimilarityConfig::new()
            .with_eps(1e-30)
            .init()
            .forward(zeros.clone(), zeros),
        [0., 0.],
    );
}

#[test]
fn cosine_dim_zero_and_clamped_norm() {
    let device = Default::default();
    let first = Tensor::<TestBackend, 2>::from_floats([[1., 0.], [0., 1.]], &device);
    let second = Tensor::<TestBackend, 2>::from_floats([[1., 1.], [0., 1.]], &device);
    close(
        CosineSimilarityConfig::new()
            .with_dim(-2)
            .with_eps(2.)
            .init()
            .forward(first, second),
        [0.25, 0.25],
    );
}

#[test]
fn pixel_shuffle_strided_input_backward() {
    let device = Default::default();
    let data = TensorData::new((0..24).map(|x| x as f32).collect::<Vec<_>>(), [1, 4, 2, 3]);
    let input = Tensor::<TestAutodiffBackend, 4>::from_data(data, &device).require_grad();
    let output = PixelShuffleConfig::new(2)
        .init()
        .forward(input.clone().swap_dims(2, 3));
    let weights = Tensor::<TestAutodiffBackend, 4>::from_data(
        TensorData::new((0..24).map(|x| x as f32).collect::<Vec<_>>(), [1, 1, 6, 4]),
        &device,
    );
    let grads = (output * weights).sum().backward();
    let mut expected = Vec::new();
    for channel in 0..4 {
        for height in 0..2 {
            for width in 0..3 {
                expected.push(((width * 2 + channel / 2) * 4 + height * 2 + channel % 2) as f32);
            }
        }
    }
    close(
        input.grad(&grads).unwrap(),
        TensorData::new(expected, [1, 4, 2, 3]),
    );
}

#[test]
fn pixel_unshuffle_backward() {
    let device = Default::default();
    let input = Tensor::<TestAutodiffBackend, 4>::ones([1, 1, 4, 4], &device).require_grad();
    let output = PixelUnshuffleConfig::new(2).init().forward(input.clone());
    let weights = Tensor::<TestAutodiffBackend, 4>::from_data(
        TensorData::new((0..16).map(|x| x as f32).collect::<Vec<_>>(), [1, 4, 2, 2]),
        &device,
    );
    let grads = (output * weights).sum().backward();
    close(
        input.grad(&grads).unwrap(),
        [[[
            [0., 4., 1., 5.],
            [8., 12., 9., 13.],
            [2., 6., 3., 7.],
            [10., 14., 11., 15.],
        ]]],
    );
}

#[test]
fn fold_padding_dilation_output_padding_forward_backward() {
    let device = Default::default();
    let columns: Vec<f32> = (0..144).map(|x| x as f32 / 16.).collect();
    let weights: Vec<f32> = (0..168).map(|x| x as f32 / 32.).collect();
    let mut expected = vec![0f32; 168];
    let mut expected_grad = vec![0f32; 144];
    for batch in 0..2 {
        for channel in 0..2 {
            for kh in 0..2 {
                for kw in 0..2 {
                    for row in 0..3 {
                        for col in 0..3 {
                            let h = row as isize * 2 + kh as isize * 2 - 1;
                            let w = col * 2 + kw;
                            if (0..6).contains(&h) && w < 7 {
                                let src =
                                    (batch * 8 + channel * 4 + kh * 2 + kw) * 9 + row * 3 + col;
                                let dst = ((batch * 2 + channel) * 6 + h as usize) * 7 + w;
                                expected[dst] += columns[src];
                                expected_grad[src] = weights[dst];
                            }
                        }
                    }
                }
            }
        }
    }
    let input =
        Tensor::<TestAutodiffBackend, 3>::from_data(TensorData::new(columns, [2, 8, 9]), &device)
            .require_grad();
    let output = Fold4dConfig::new([6, 7], [2, 2])
        .with_stride([2, 2])
        .with_padding([1, 0])
        .with_dilation([2, 1])
        .init()
        .forward(input.clone());
    close(
        output.clone().inner(),
        TensorData::new(expected, [2, 2, 6, 7]),
    );
    let weights = Tensor::<TestAutodiffBackend, 4>::from_data(
        TensorData::new(weights, [2, 2, 6, 7]),
        &device,
    );
    let grads = (output * weights).sum().backward();
    close(
        input.grad(&grads).unwrap(),
        TensorData::new(expected_grad, [2, 8, 9]),
    );
}

#[test]
fn fold_half_storage_and_backward() {
    let device = Default::default();
    for dtype in [DType::F16, DType::BF16] {
        let input = Tensor::<TestAutodiffBackend, 3>::ones([1, 4, 4], &device)
            .cast(dtype)
            .require_grad();
        let output = Fold4dConfig::new([3, 3], [2, 2])
            .init()
            .forward(input.clone());
        assert_eq!(output.dtype(), dtype);
        close(
            output.clone().inner(),
            [[[[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]]],
        );
        close(
            input.grad(&output.sum().backward()).unwrap(),
            [[[1., 1., 1., 1.]; 4]],
        );
    }
}

#[test]
fn all_loss_auto_reductions() {
    let device = Default::default();
    let x = Tensor::<TestBackend, 1>::from_floats([1., 2.], &device);
    let y = Tensor::<TestBackend, 1>::zeros([2], &device);
    let ones = Tensor::<TestBackend, 1>::ones([2], &device);
    close(
        GaussianNLLLossConfig::new().init().forward(
            x.clone(),
            y.clone(),
            ones.clone(),
            Reduction::Auto,
        ),
        [1.25],
    );
    close(
        HingeEmbeddingLossConfig::new()
            .init()
            .forward(x.clone(), ones.clone(), Reduction::Auto),
        [1.5],
    );
    close(
        MarginRankingLossConfig::new()
            .init()
            .forward(y, x.clone(), ones.clone(), Reduction::Auto),
        [1.5],
    );
    close(
        SoftMarginLoss::new().forward(x, ones, Reduction::Auto),
        [0.22009485],
    );
    close(
        MultiMarginLossConfig::new().init().forward(
            Tensor::from_floats([[1., 2.]], &device),
            Tensor::from_ints([0], &device),
            Reduction::Auto,
        ),
        [1.],
    );
    close(
        TripletMarginLossConfig::new().init().forward(
            Tensor::from_floats([[0., 0.]], &device),
            Tensor::from_floats([[3., 4.]], &device),
            Tensor::from_floats([[0., 2.]], &device),
            Reduction::Auto,
        ),
        [4.],
    );
}

#[test]
#[should_panic(expected = "input channels")]
fn fold_rejects_incompatible_channels() {
    Fold4dConfig::new([3, 3], [2, 2])
        .init()
        .forward(Tensor::<TestBackend, 3>::zeros(
            [1, 5, 4],
            &Default::default(),
        ));
}

#[test]
#[should_panic(expected = "number of blocks")]
fn fold_rejects_incompatible_block_grid() {
    Fold4dConfig::new([3, 3], [2, 2])
        .init()
        .forward(Tensor::<TestBackend, 3>::zeros(
            [1, 4, 5],
            &Default::default(),
        ));
}

#[test]
#[should_panic(expected = "upscale_factor must be greater than 0")]
fn pixel_shuffle_rejects_zero_factor() {
    PixelShuffleConfig::new(0)
        .init()
        .forward(Tensor::<TestBackend, 4>::zeros(
            [1, 4, 2, 2],
            &Default::default(),
        ));
}

#[test]
#[should_panic(expected = "input width")]
fn pixel_unshuffle_rejects_incompatible_width() {
    PixelUnshuffleConfig::new(2)
        .init()
        .forward(Tensor::<TestBackend, 4>::zeros(
            [1, 1, 2, 3],
            &Default::default(),
        ));
}

#[test]
#[should_panic(expected = "non-negative")]
fn margin_ranking_rejects_negative_margin() {
    MarginRankingLossConfig::new().with_margin(-1.).init();
}

#[test]
#[should_panic(expected = "p = 1 or p = 2")]
fn multi_margin_rejects_invalid_exponent() {
    MultiMarginLossConfig::new().with_p(3).init();
}

#[test]
#[should_panic(expected = "outside")]
fn cosine_rejects_invalid_dimension() {
    let x = Tensor::<TestBackend, 2>::ones([1, 2], &Default::default());
    CosineSimilarityConfig::new()
        .with_dim(-3)
        .init()
        .forward(x.clone(), x);
}
