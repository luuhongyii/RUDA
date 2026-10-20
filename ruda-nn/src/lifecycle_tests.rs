use crate::{
    BatchNormConfig, Dropout, DropoutConfig, GaussianNoiseConfig, Linear, LinearConfig,
    RReluConfig, TestAutodiffBackend as AD, TestBackend as B,
    activation::{Activation, ActivationConfig},
    attention::{
        DenseAttentionMask, PackedSequenceLayout, dense_scaled_dot_product_attention,
        packed_scaled_dot_product_attention,
    },
};
use alloc::{vec, vec::Vec};
use ruda_model::{
    module::{AutodiffModule, Flag, Module, ModuleVisitor, Param, ParamGroup},
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
    tensor::{Tensor, TensorData, Tolerance, backend::Backend},
};

fn close(actual: TensorData, expected: TensorData) {
    actual.assert_approx_eq::<f32>(&expected, Tolerance::rel_abs(1e-5, 1e-5));
}

#[derive(Clone, Debug)]
struct Opaque(usize);
type DropAlias = Dropout;

#[derive(Module, Clone, Debug)]
struct GenericConstant<T: Clone + core::fmt::Debug + Send> {
    #[module(skip)]
    value: T,
    training: Param<Flag>,
}

#[derive(Module, Clone, Debug)]
struct ConstantTree {
    layers: Vec<Option<DropAlias>>,
    array: [Dropout; 1],
    pair: (Dropout, Dropout),
    opaque: Opaque,
    #[module(skip)]
    skipped: Dropout,
}

#[derive(Module, Clone, Debug)]
enum ConstantEnum {
    Tree(ConstantTree),
}

#[derive(Module, Debug)]
struct Nested<B: Backend> {
    linear: Linear<B>,
    controls: ConstantEnum,
    activation: Activation<B>,
}

fn tree() -> ConstantTree {
    ConstantTree {
        layers: vec![
            Some(DropoutConfig::new(0.5).init()),
            None,
            Some(DropoutConfig::new(0.2).init()),
        ],
        array: [DropoutConfig::new(0.5).init()],
        pair: (
            DropoutConfig::new(0.5).init(),
            DropoutConfig::new(0.5).init(),
        ),
        opaque: Opaque(17),
        skipped: DropoutConfig::new(0.5).init(),
    }
}

fn nested() -> Nested<AD> {
    Nested {
        linear: LinearConfig::new(2, 2).init(&Default::default()),
        controls: ConstantEnum::Tree(tree()),
        activation: ActivationConfig::RRelu(RReluConfig::new()).init(&Default::default()),
    }
}

#[test]
fn flag_validation_restores_state_and_identity() {
    let flag = Param::<Flag>::from_bool(true);
    let id = flag.id;
    let valid = AutodiffModule::<AD>::valid(&flag);
    assert!(!valid.is_enabled());
    let mapped = valid.map(|f| f.with(false));
    let train = <Param<Flag> as AutodiffModule<AD>>::from_inner(mapped);
    assert!(train.is_enabled());
    assert_eq!(train.id, id);
    let off = AutodiffModule::<AD>::valid(&train.with_value(false));
    assert!(!<Param<Flag> as AutodiffModule<AD>>::from_inner(off.clone()).is_enabled());
    assert!(<Param<Flag> as AutodiffModule<AD>>::from_inner(off.with_value(true)).is_enabled());
}

#[test]
fn flag_record_device_and_no_grad_preserve_configuration() {
    let source = Param::<Flag>::from_bool(true);
    let destination = Param::<Flag>::from_bool(false);
    let id = destination.id;
    let loaded = Module::<B>::load_record(destination, Module::<B>::into_record(source));
    let moved = Module::<B>::to_device(loaded, &Default::default());
    let forked = Module::<B>::fork(moved, &Default::default());
    assert_eq!(forked.id, id);
    assert!(!forked.is_enabled());
    assert!(Module::<B>::devices(&forked).is_empty());
    assert!(Module::<B>::no_grad(forked.with_value(true)).is_enabled());
}

#[test]
fn nested_freeze_and_validation_cover_constant_containers() {
    let generic = GenericConstant {
        value: Opaque(29),
        training: Param::from_bool(true),
    };
    let generic = Module::<AD>::freeze(generic);
    assert!(!generic.training.is_enabled());
    assert_eq!(generic.value.0, 29);
    let model = nested();
    assert_eq!(model.num_params(), 6);
    let frozen = model.freeze();
    let ConstantEnum::Tree(ref controls) = frozen.controls;
    assert!(!controls.layers[0].as_ref().unwrap().training.is_enabled());
    assert!(!controls.layers[2].as_ref().unwrap().training.is_enabled());
    assert!(!controls.array[0].training.is_enabled());
    assert!(!controls.pair.0.training.is_enabled());
    assert!(!controls.pair.1.training.is_enabled());
    assert!(controls.skipped.training.is_enabled());
    assert_eq!(controls.opaque.0, 17);
    assert!(!frozen.linear.weight.is_require_grad());
    let valid = frozen.unfreeze().valid();
    let ConstantEnum::Tree(ref controls) = valid.controls;
    assert!(!controls.array[0].training.is_enabled());
    let trained = valid.train::<AD>();
    let ConstantEnum::Tree(ref controls) = trained.controls;
    assert!(controls.array[0].training.is_enabled());
    assert!(trained.linear.weight.is_require_grad());
    assert!(matches!(trained.activation, Activation::RRelu(ref a) if a.training.is_enabled()));
}

#[test]
fn grouped_freeze_uses_nested_paths_and_flag_ids() {
    let model = nested().freeze_group(
        ParamGroup::from_path("controls.Tree.layers.2.training")
            .fuse(&ParamGroup::from_path("linear.weight")),
    );
    let ConstantEnum::Tree(ref controls) = model.controls;
    assert!(controls.layers[0].as_ref().unwrap().training.is_enabled());
    assert!(!controls.layers[2].as_ref().unwrap().training.is_enabled());
    assert!(!model.linear.weight.is_require_grad());
    assert!(model.linear.bias.as_ref().unwrap().is_require_grad());
    let id = controls.layers[2].as_ref().unwrap().training.id;
    let model = model.unfreeze_group(ParamGroup::from_ids(vec![id]));
    let ConstantEnum::Tree(ref controls) = model.controls;
    assert!(controls.layers[2].as_ref().unwrap().training.is_enabled());
    assert!(!model.linear.weight.is_require_grad());
}

#[test]
fn subtree_ids_include_flags_without_initializing_lazy_weights() {
    let model = nested();
    assert!(!model.linear.weight.is_initialized());
    let group = ParamGroup::ids_from_module(model.clone());
    assert!(!model.linear.weight.is_initialized());
    assert!(group.matches(&model.linear.weight.id, None));
    let ConstantEnum::Tree(controls) = model.controls;
    assert!(group.matches(&controls.array[0].training.id, None));
    assert!(!group.matches(&controls.skipped.training.id, None));
}

#[test]
fn unfreeze_on_inner_backend_restores_gradients_on_train() {
    let model = nested().freeze();
    let inner = model
        .valid()
        .unfreeze_group(ParamGroup::from_path("linear.weight"));
    assert!(!inner.linear.weight.is_require_grad());
    let train = inner.train::<AD>();
    assert!(train.linear.weight.is_require_grad());
    assert!(!train.linear.bias.as_ref().unwrap().is_require_grad());
    let ConstantEnum::Tree(controls) = train.controls;
    assert!(!controls.array[0].training.is_enabled());
}

#[test]
fn group_predicates_union_exclusion_and_empty_lists() {
    let id = Param::<Flag>::from_bool(true).id;
    let group = ParamGroup::from_predicates(vec!["encoder", "weight"])
        .fuse(&ParamGroup::from_any_predicates(vec![
            "decoder", "training",
        ]))
        .exclude(ParamGroup::from_path("decoder.bias"));
    assert!(group.matches(&id, Some("encoder.layer.weight")));
    assert!(group.matches(&id, Some("block.training")));
    assert!(!group.matches(&id, Some("encoder.bias")));
    assert!(!group.matches(&id, Some("decoder.bias")));
    assert!(!group.matches(&id, None));
    assert!(ParamGroup::from_any_predicates(Vec::<&str>::new()).matches(&id, Some("")));
    assert!(!ParamGroup::from_paths(Vec::<&str>::new()).matches(&id, Some("")));
    assert!(
        ParamGroup::all()
            .exclude(ParamGroup::from_ids(vec![id]))
            .fuse(&ParamGroup::all())
            .matches(&id, None)
    );
}

#[cfg(feature = "std")]
#[test]
fn group_regex_and_invalid_patterns() {
    let id = Param::<Flag>::from_bool(true).id;
    let group = ParamGroup::from_regexes(vec![r"^layers\.\d+\.", "training$"]).unwrap();
    assert!(group.matches(&id, Some("layers.2.training")));
    assert!(!group.matches(&id, Some("layers.2.weight")));
    let group = ParamGroup::from_any_regexes(vec!["bias$", "training$"]).unwrap();
    assert!(group.matches(&id, Some("linear.bias")));
    assert!(group.matches(&id, Some("dropout.training")));
    assert!(ParamGroup::from_regex("[").is_err());
}

#[test]
fn rrelu_inference_midpoint_and_equal_bounds() {
    let input = Tensor::<B, 1>::from_data([-2., 0., 3.], &Default::default());
    let layer = RReluConfig::new().with_lower(0.1).with_upper(0.3).init();
    close(
        layer.forward(input).to_data(),
        TensorData::from([-0.4, 0., 3.]),
    );
    let input = Tensor::<AD, 1>::from_data([-2., 0., 3.], &Default::default());
    close(
        RReluConfig::new()
            .with_lower(0.25)
            .with_upper(0.25)
            .init()
            .forward(input)
            .to_data(),
        TensorData::from([-0.5, 0., 3.]),
    );
}

#[test]
#[should_panic(expected = "lower <= upper")]
fn rrelu_invalid_bounds() {
    RReluConfig::new().with_lower(0.5).with_upper(0.1).init();
}

#[test]
fn rrelu_training_gradients_match_sampled_slopes() {
    let input = Tensor::<AD, 1>::full([128], -1.0, &Default::default()).require_grad();
    let output = RReluConfig::new()
        .with_lower(0.1)
        .with_upper(0.3)
        .init()
        .forward(input.clone());
    let values = output.to_data().to_vec::<f32>().unwrap();
    assert!(values.iter().all(|v| *v <= -0.1 && *v >= -0.3));
    assert!(values.windows(2).any(|p| p[0] != p[1]));
    let grads = output.clone().sum().backward();
    close(
        input.grad(&grads).unwrap().to_data(),
        output.neg().to_data(),
    );
}

#[test]
fn rrelu_frozen_autodiff_keeps_input_gradient() {
    let layer = Module::<AD>::freeze(RReluConfig::new().with_lower(0.1).with_upper(0.3).init());
    let input = Tensor::<AD, 1>::from_data([-2., 1.], &Default::default()).require_grad();
    let output = layer.forward(input.clone());
    close(output.to_data(), TensorData::from([-0.4, 1.]));
    let grads = output.sum().backward();
    close(
        input.grad(&grads).unwrap().to_data(),
        TensorData::from([0.2, 1.]),
    );
}

#[test]
fn activation_wrapper_preserves_rrelu_lifecycle() {
    let layer: Activation<AD> =
        ActivationConfig::RRelu(RReluConfig::new()).init(&Default::default());
    let valid = layer.valid();
    assert!(matches!(valid, Activation::RRelu(ref a) if !a.training.is_enabled()));
    let train = valid.train::<AD>();
    assert!(matches!(train, Activation::RRelu(ref a) if a.training.is_enabled()));
    let frozen = train.freeze();
    close(
        frozen
            .forward(Tensor::<AD, 1>::from_data([-1.], &Default::default()))
            .to_data(),
        TensorData::from([-((0.125 + 1.0 / 3.0) / 2.0) as f32]),
    );
}

#[test]
fn no_grad_preserves_all_stochastic_training_flags() {
    let dropout = Module::<AD>::no_grad(DropoutConfig::new(1.0).init());
    let noise = Module::<AD>::no_grad(GaussianNoiseConfig::new(0.5).init());
    let rrelu = Module::<AD>::no_grad(RReluConfig::new().init());
    let input = Tensor::<AD, 1>::ones([128], &Default::default());
    close(
        dropout.forward(input.clone()).to_data(),
        TensorData::new(vec![0.0f32; 128], [128]),
    );
    assert_ne!(noise.forward(input.clone()).to_data(), input.to_data());
    let output = rrelu
        .forward(input.neg())
        .to_data()
        .to_vec::<f32>()
        .unwrap();
    assert!(output.windows(2).any(|p| p[0] != p[1]));
}

#[test]
fn frozen_dropout_and_noise_are_identity_on_autodiff() {
    let input = Tensor::<AD, 1>::from_data([1., -2., 3.], &Default::default()).require_grad();
    let dropout = Module::<AD>::freeze(DropoutConfig::new(1.0).init());
    let noise = Module::<AD>::freeze(GaussianNoiseConfig::new(0.5).init());
    let output = noise.forward(dropout.forward(input.clone()));
    assert_eq!(output.to_data(), input.to_data());
    let grads = output.sum().backward();
    close(
        input.grad(&grads).unwrap().to_data(),
        TensorData::from([1., 1., 1.]),
    );
    assert!(Module::<AD>::unfreeze(dropout).training.is_enabled());
    assert!(Module::<AD>::unfreeze(noise).training.is_enabled());
}

#[test]
fn batchnorm_freeze_stops_statistics_but_no_grad_does_not() {
    let device = Default::default();
    let layer = BatchNormConfig::new(1)
        .with_momentum(0.5)
        .init::<AD>(&device)
        .freeze();
    let input = Tensor::<AD, 2>::from_data([[2.], [4.]], &device).require_grad();
    let before = layer.running_mean.value_sync().to_data();
    let output = layer.forward(input.clone());
    assert_eq!(before, layer.running_mean.value_sync().to_data());
    close(
        output.to_data(),
        TensorData::from([[2. / (1.0f32 + 1e-5).sqrt()], [4. / (1.0f32 + 1e-5).sqrt()]]),
    );
    assert!(input.grad(&output.sum().backward()).is_some());
    let layer = layer.unfreeze().no_grad();
    assert!(!layer.gamma.is_require_grad());
    layer.forward(input);
    close(
        layer.running_mean.value_sync().to_data(),
        TensorData::from([1.5]),
    );
}

#[test]
fn batchnorm_records_omit_flags_and_keep_destination_state() {
    let device = Default::default();
    let source = BatchNormConfig::new(1).init::<AD>(&device);
    source.gamma.val();
    source.beta.val();
    let mut off = source.clone();
    off.training = off.training.with_value(false);
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    let bytes = Recorder::<AD>::record(&recorder, source.clone().into_record(), ()).unwrap();
    let off_bytes = Recorder::<AD>::record(&recorder, off.into_record(), ()).unwrap();
    assert_eq!(bytes, off_bytes);
    let mut destination = BatchNormConfig::new(1).init::<AD>(&device).valid();
    let id = destination.training.id;
    destination = destination.load_record(source.valid().into_record());
    assert_eq!(destination.training.id, id);
    assert!(!destination.training.is_enabled());
    assert!(destination.train::<AD>().training.is_enabled());
}

#[test]
fn frozen_probability_one_dropout_does_not_zero_attention() {
    let device = Default::default();
    let dropout = Module::<AD>::freeze(DropoutConfig::new(1.0).init());
    let q = Tensor::<AD, 4>::ones([1, 1, 1, 1], &device);
    let v = Tensor::<AD, 4>::full([1, 1, 1, 1], 3., &device).require_grad();
    let output = dense_scaled_dot_product_attention(
        q.clone(),
        q,
        v.clone(),
        DenseAttentionMask::default(),
        Default::default(),
        Some(&dropout),
    );
    close(output.to_data(), TensorData::from([[[[3.]]]]));
    close(
        v.grad(&output.sum().backward()).unwrap().to_data(),
        TensorData::from([[[[1.]]]]),
    );
    let q = Tensor::<AD, 3>::ones([1, 1, 1], &device);
    let v = Tensor::<AD, 3>::full([1, 1, 1], 3., &device).require_grad();
    let layout = PackedSequenceLayout::new(vec![0, 1], 1);
    let output = packed_scaled_dot_product_attention(
        q.clone(),
        q,
        v.clone(),
        &layout,
        &layout,
        Default::default(),
        Some(&dropout),
    );
    close(output.to_data(), TensorData::from([[[3.]]]));
    close(
        v.grad(&output.sum().backward()).unwrap().to_data(),
        TensorData::from([[[1.]]]),
    );
}

#[test]
fn frozen_display_exposes_training_state() {
    let layer = Module::<AD>::freeze(RReluConfig::new().with_lower(0.1).with_upper(0.3).init());
    assert_eq!(
        alloc::format!("{layer}"),
        "RRelu {lower: 0.1, upper: 0.3, training: disabled}"
    );
}

#[test]
fn visitor_reaches_each_flag_once() {
    #[derive(Default)]
    struct Visitor(usize);
    impl ModuleVisitor<AD> for Visitor {
        fn visit_flag(&mut self, _: &Param<Flag>) {
            self.0 += 1;
        }
    }
    let mut visitor = Visitor::default();
    nested().visit(&mut visitor);
    assert_eq!(visitor.0, 6);
}
