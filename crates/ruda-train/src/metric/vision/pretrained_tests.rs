use super::{AfineConfig, DistsConfig, FidConfig, LpipsConfig, LpipsNet};
use crate::tests::TestBackend;
use ruda_model::tensor::TensorData;
use ruda_store::pytorch::PytorchReader;
use ruda_store::{ModuleSnapshot, TensorSnapshot};
use std::collections::BTreeMap;
use std::path::PathBuf;

fn cache(metric: &str, file: &str) -> PathBuf {
    dirs::cache_dir()
        .unwrap()
        .join("burn-dataset")
        .join(metric)
        .join(file)
}

fn tensors(module: &impl ModuleSnapshot<TestBackend>) -> BTreeMap<String, TensorSnapshot> {
    module
        .collect(None, None, true)
        .into_iter()
        .map(|t| (t.full_path(), t))
        .collect()
}

fn compare_data(
    name: &str,
    source: TensorData,
    actual: TensorData,
    transpose: bool,
    flatten: bool,
) {
    let mut shape = source.shape.clone();
    if transpose {
        assert_eq!(shape.len(), 2, "{name}");
        shape.swap(0, 1);
    }
    if flatten {
        shape = [shape.iter().product::<usize>()].into();
    }
    assert_eq!(actual.shape, shape, "{name}");
    let expected = source.convert::<f32>().to_vec::<f32>().unwrap();
    let actual = actual.convert::<f32>().to_vec::<f32>().unwrap();
    assert_eq!(actual.len(), expected.len(), "{name}");
    for (index, value) in actual.iter().enumerate() {
        let source_index = if transpose {
            (index % shape[1]) * shape[0] + index / shape[1]
        } else {
            index
        };
        assert_eq!(*value, expected[source_index], "{name}[{index}]");
    }
}

fn check(
    actual: &mut BTreeMap<String, TensorSnapshot>,
    target: &str,
    reader: &PytorchReader,
    source: &str,
    transpose: bool,
    flatten: bool,
) {
    let loaded = actual
        .remove(target)
        .unwrap_or_else(|| panic!("Missing target {target}"));
    let expected = reader
        .get(source)
        .unwrap_or_else(|| panic!("Missing source {source}"));
    compare_data(
        target,
        expected.to_data().unwrap(),
        loaded.to_data().unwrap(),
        transpose,
        flatten,
    );
}

const VGG_LAYERS: [(usize, &str); 13] = [
    (0, "conv1_1"),
    (2, "conv1_2"),
    (5, "conv2_1"),
    (7, "conv2_2"),
    (10, "conv3_1"),
    (12, "conv3_2"),
    (14, "conv3_3"),
    (17, "conv4_1"),
    (19, "conv4_2"),
    (21, "conv4_3"),
    (24, "conv5_1"),
    (26, "conv5_2"),
    (28, "conv5_3"),
];

fn check_conv(
    actual: &mut BTreeMap<String, TensorSnapshot>,
    reader: &PytorchReader,
    index: usize,
    field: &str,
) {
    for suffix in ["weight", "bias"] {
        check(
            actual,
            &format!("extractor.{field}.{suffix}"),
            reader,
            &format!("features.{index}.{suffix}"),
            false,
            false,
        );
    }
}

fn lpips_weights(net: LpipsNet, file: &str, count: usize) {
    let metric = LpipsConfig::new()
        .with_net(net)
        .init_pretrained::<TestBackend>(&Default::default());
    let mut actual = tensors(&metric);
    assert_eq!(actual.len(), count);
    let backbone = PytorchReader::new(cache("lpips", &format!("{file}_backbone.pth"))).unwrap();
    let linear = PytorchReader::new(cache("lpips", &format!("{file}_lpips.pth"))).unwrap();
    match net {
        LpipsNet::Vgg => {
            for (index, field) in VGG_LAYERS {
                check_conv(&mut actual, &backbone, index, field);
            }
        }
        LpipsNet::Alex => {
            for (i, index) in [0, 3, 6, 8, 10].into_iter().enumerate() {
                check_conv(&mut actual, &backbone, index, &format!("conv{}", i + 1));
            }
        }
        LpipsNet::Squeeze => {
            check_conv(&mut actual, &backbone, 0, "conv1");
            for (i, index) in [3, 4, 6, 7, 9, 10, 11, 12].into_iter().enumerate() {
                for conv in ["squeeze", "expand1x1", "expand3x3"] {
                    for suffix in ["weight", "bias"] {
                        check(
                            &mut actual,
                            &format!("extractor.fire{}.{conv}.{suffix}", i + 1),
                            &backbone,
                            &format!("features.{index}.{conv}.{suffix}"),
                            false,
                            false,
                        );
                    }
                }
            }
        }
    }
    for index in 0..if net == LpipsNet::Squeeze { 7 } else { 5 } {
        check(
            &mut actual,
            &format!("lin{index}.weight"),
            &linear,
            &format!("lin{index}.model.1.weight"),
            false,
            false,
        );
    }
    assert!(
        actual.is_empty(),
        "Unchecked: {:?}",
        actual.keys().collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "downloads pre-trained weights"]
fn lpips_vgg_all_pretrained_weights() {
    lpips_weights(LpipsNet::Vgg, "vgg", 31);
}

#[test]
#[ignore = "downloads pre-trained weights"]
fn lpips_alex_all_pretrained_weights() {
    lpips_weights(LpipsNet::Alex, "alex", 15);
}

#[test]
#[ignore = "downloads pre-trained weights"]
fn lpips_squeeze_all_pretrained_weights() {
    lpips_weights(LpipsNet::Squeeze, "squeeze", 57);
}

fn tar_matches_pytorch(net: &str, count: usize) {
    let reader = PytorchReader::new(cache("lpips", &format!("{net}_backbone.pth"))).unwrap();
    let reference =
        PytorchReader::new(cache("lpips", &format!("{net}_pytorch_reference.pth"))).unwrap();
    let mut keys = reader.keys();
    let mut expected_keys = reference.keys();
    keys.sort();
    expected_keys.sort();
    assert_eq!(keys, expected_keys);
    assert_eq!(keys.len(), count);
    for name in keys {
        let actual = reader.get(&name).unwrap().to_data().unwrap();
        let expected = reference.get(&name).unwrap().to_data().unwrap();
        assert_eq!(actual.dtype, expected.dtype, "{name}");
        compare_data(&name, expected, actual, false, false);
    }
}

#[test]
#[ignore = "requires official weights and a PyTorch-generated ZIP reference"]
fn alexnet_tar_matches_pytorch() {
    tar_matches_pytorch("alex", 16);
}

#[test]
#[ignore = "requires official weights and a PyTorch-generated ZIP reference"]
fn squeezenet_tar_matches_pytorch() {
    tar_matches_pytorch("squeeze", 52);
}

#[test]
#[ignore = "downloads pre-trained weights"]
fn dists_all_pretrained_weights() {
    let metric = DistsConfig::new().init_pretrained::<TestBackend>(&Default::default());
    let mut actual = tensors(&metric);
    assert_eq!(actual.len(), 32);
    for (index, channels) in [64, 128, 256, 512].into_iter().enumerate() {
        let path = format!("extractor.pool{}.conv.weight", index + 1);
        let filter = actual.remove(&path).unwrap().to_data().unwrap();
        assert_eq!(filter.shape.as_slice(), [channels, 1, 5, 5]);
        let values = filter.to_vec::<f32>().unwrap();
        let window = [0.0f32, 0.5, 1.0, 0.5, 0.0];
        for (i, value) in values.iter().enumerate() {
            let expected = window[i % 25 / 5] * window[i % 5] / 4.0;
            assert!((value - expected).abs() < 1e-7, "{path}[{i}]");
        }
    }
    let backbone = PytorchReader::new(cache("dists", "vgg16_backbone.pth")).unwrap();
    for (index, field) in VGG_LAYERS {
        check_conv(&mut actual, &backbone, index, field);
    }
    let learned = PytorchReader::new(cache("dists", "dists_weights.pt")).unwrap();
    for name in ["alpha", "beta"] {
        check(&mut actual, name, &learned, name, false, true);
    }
    assert!(actual.is_empty());
}

#[test]
#[ignore = "downloads pre-trained weights"]
fn fid_all_pretrained_weights() {
    let metric = FidConfig::new().init_pretrained::<TestBackend>(&Default::default());
    let mut actual = tensors(&metric);
    assert_eq!(actual.len(), 470);
    let reader = PytorchReader::new(cache("fid", "pt-inception-2015-12-05-6726825d.pth")).unwrap();
    let mut checked = 0;
    for source in reader.keys() {
        let Some((module, suffix)) = source.split_once('.') else {
            continue;
        };
        if suffix.ends_with("num_batches_tracked") {
            continue;
        }
        let field = match module {
            "Conv2d_1a_3x3" => "conv2d_1a".to_owned(),
            "Conv2d_2a_3x3" => "conv2d_2a".to_owned(),
            "Conv2d_2b_3x3" => "conv2d_2b".to_owned(),
            "Conv2d_3b_1x1" => "conv2d_3b".to_owned(),
            "Conv2d_4a_3x3" => "conv2d_4a".to_owned(),
            name if name.starts_with("Mixed_") => name.to_lowercase(),
            _ => continue,
        };
        let target = format!("extractor.{field}.{suffix}")
            .replace(".bn.weight", ".bn.gamma")
            .replace(".bn.bias", ".bn.beta");
        check(&mut actual, &target, &reader, &source, false, false);
        checked += 1;
    }
    assert_eq!(checked, 470);
    assert!(actual.is_empty());
}

#[test]
#[ignore = "downloads pre-trained weights"]
fn afine_all_pretrained_weights() {
    let metric = AfineConfig::new()
        .with_image_size(224)
        .init_pretrained::<TestBackend>(&Default::default());
    let mut actual = tensors(&metric);
    assert_eq!(actual.len(), 164);
    let path = cache("afine", "afine.pth");
    let clip = PytorchReader::with_top_level_key(&path, "finetuned_clip").unwrap();
    let mut visual_count = 0;
    for source in clip
        .keys()
        .into_iter()
        .filter(|s| s.starts_with("visual.") && s != "visual.proj")
    {
        let suffix = source.strip_prefix("visual.").unwrap();
        let target = suffix
            .replacen("conv1.", "patch_embed.", 1)
            .replace("class_embedding", "class_token")
            .replace("transformer.resblocks.", "blocks.")
            .replace("attn.in_proj_weight", "attn.qkv_proj.weight")
            .replace("attn.in_proj_bias", "attn.qkv_proj.bias");
        let mut target = format!("clip_visual.{target}");
        for norm in ["ln_pre", "ln_post", "ln_1", "ln_2"] {
            target = target
                .replace(&format!("{norm}.weight"), &format!("{norm}.gamma"))
                .replace(&format!("{norm}.bias"), &format!("{norm}.beta"));
        }
        let transpose = clip.get(&source).unwrap().shape.len() == 2 && source.ends_with("weight");
        check(&mut actual, &target, &clip, &source, transpose, false);
        visual_count += 1;
    }
    assert_eq!(visual_count, 151);
    for (shard, prefix) in [
        ("natural", "qhead"),
        ("fidelity", "dhead"),
        ("natural_scale", "nr_calibrator"),
        ("fidelity_scale", "fr_calibrator"),
        ("adapter", "adapter"),
    ] {
        let reader = PytorchReader::with_top_level_key(&path, shard).unwrap();
        for source in reader.keys() {
            let data = reader.get(&source).unwrap().to_data().unwrap();
            if source == "mean" || source == "std" {
                let tensor = match (shard, source.as_str()) {
                    ("natural", "mean") => metric.qhead.mean.clone(),
                    ("natural", "std") => metric.qhead.std.clone(),
                    ("fidelity", "mean") => metric.dhead.mean.clone(),
                    ("fidelity", "std") => metric.dhead.std.clone(),
                    _ => panic!("Unexpected buffer {shard}.{source}"),
                };
                compare_data(
                    &format!("{shard}.{source}"),
                    data,
                    tensor.into_data(),
                    false,
                    false,
                );
                continue;
            }
            let target = format!(
                "{prefix}.{}",
                source
                    .replace("proj_head.0.", "proj_head_fc1.")
                    .replace("proj_head.2.", "proj_head_fc2.")
            );
            check(
                &mut actual,
                &target,
                &reader,
                &source,
                data.shape.len() == 2 && source.ends_with("weight"),
                data.shape.is_empty(),
            );
        }
    }
    assert!(
        actual.is_empty(),
        "Unchecked: {:?}",
        actual.keys().collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "downloads pre-trained weights"]
fn afine_pretrained_position_embedding_256() {
    let metric = AfineConfig::new().init_pretrained::<TestBackend>(&Default::default());
    let actual = metric.clip_visual.positional_embedding.val().into_data();
    assert_eq!(actual.shape.as_slice(), [65, 768]);
    let actual = actual.convert::<f32>().to_vec::<f32>().unwrap();
    let reader =
        PytorchReader::with_top_level_key(cache("afine", "afine.pth"), "finetuned_clip").unwrap();
    let source = reader
        .get("visual.positional_embedding")
        .unwrap()
        .to_data()
        .unwrap();
    assert_eq!(source.shape.as_slice(), [50, 768]);
    let source = source.convert::<f32>().to_vec::<f32>().unwrap();
    let cubic = |distance: f64| {
        let x = distance.abs();
        if x <= 1.0 {
            (1.25 * x - 2.25) * x * x + 1.0
        } else if x < 2.0 {
            ((-0.75 * x + 3.75) * x - 6.0) * x + 3.0
        } else {
            0.0
        }
    };
    for row in 0..65 {
        let position = (row as f64 + 0.5) * 50.0 / 65.0 - 0.5;
        let base = position.floor() as i64;
        for column in 0..768 {
            let expected: f64 = (-1..=2)
                .map(|offset| {
                    let index = base + offset;
                    source[index.clamp(0, 49) as usize * 768 + column] as f64
                        * cubic(position - index as f64)
                })
                .sum();
            let value = actual[row * 768 + column] as f64;
            assert!(
                (value - expected).abs() < 2e-5,
                "position[{row},{column}]: expected {expected}, got {value}"
            );
        }
    }
}
