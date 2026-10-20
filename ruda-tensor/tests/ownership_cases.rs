macro_rules! ownership_cases {
    ($backend:ty) => {
        use ruda_tensor::{
            Backend,
            api::{Bool, DType, Int, Tensor, TensorData, quantization::QuantScheme},
        };
        type B = $backend;

        #[test]
        fn ownership_tracks_float_int_bool_clones() {
            let device = Default::default();
            macro_rules! check {
                ($tensor:expr) => {{
                    let tensor = $tensor;
                    B::sync(&device).unwrap();
                    assert!(tensor.can_mut());
                    let alias = tensor.clone();
                    assert!(!tensor.can_mut());
                    assert!(!alias.can_mut());
                    drop(alias);
                    assert!(tensor.can_mut());
                }};
            }
            check!(Tensor::<B, 1>::from_floats([1., 2.], &device));
            check!(Tensor::<B, 1, Int>::from_data([1, 2], &device));
            check!(Tensor::<B, 1, Bool>::from_data([true, false], &device));
        }

        #[test]
        fn extract_preserves_value_and_leaves_default_dtype_placeholder() {
            let device = Default::default();
            let mut tensor =
                Tensor::<B, 2>::from_data(TensorData::new(vec![1.25f64, -2.5], [1, 2]), &device)
                    .cast(DType::F64);
            let extracted = tensor.extract();
            assert_eq!(extracted.dims(), [1, 2]);
            assert_eq!(extracted.dtype(), DType::F64);
            assert_eq!(extracted.device(), device);
            assert_eq!(extracted.into_data().to_vec::<f64>().unwrap(), [1.25, -2.5]);
            assert_eq!(tensor.dims(), [0, 0]);
            assert_eq!(tensor.dtype(), DType::F32);
            assert_eq!(tensor.device(), device);
            assert_eq!(tensor.extract().dims(), [0, 0]);
        }

        #[test]
        fn inplace_uses_owned_operation_without_changing_alias() {
            let device = Default::default();
            let mut tensor = Tensor::<B, 1>::from_floats([2., -3.], &device);
            let alias = tensor.clone();
            tensor.inplace(|value| value.square());
            assert_eq!(tensor.into_data().to_vec::<f32>().unwrap(), [4., 9.]);
            assert_eq!(alias.into_data().to_vec::<f32>().unwrap(), [2., -3.]);
        }

        #[test]
        fn inplace_panic_leaves_reusable_empty_placeholder() {
            let device = Default::default();
            let mut tensor = Tensor::<B, 1>::from_floats([2., -3.], &device);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                tensor.inplace(|_| panic!("owned operation failed"));
            }));
            assert!(result.is_err());
            assert_eq!(tensor.dims(), [0]);
            assert_eq!(tensor.device(), device);
            tensor.inplace(|_| Tensor::from_floats([7.], &device));
            assert_eq!(tensor.into_data().to_vec::<f32>().unwrap(), [7.]);
        }

        #[test]
        fn empty_like_preserves_metadata_including_zero_dimensions() {
            let device = Default::default();
            macro_rules! check {
                ($tensor:expr) => {{
                    let tensor = $tensor;
                    let empty = tensor.empty_like();
                    assert_eq!(empty.dims(), tensor.dims());
                    assert_eq!(empty.dtype(), tensor.dtype());
                    assert_eq!(empty.device(), tensor.device());
                }};
            }
            check!(Tensor::<B, 2>::from_floats([[1., 2.]], &device).cast(DType::F64));
            check!(Tensor::<B, 2, Int>::from_data([[1, 2]], &device).cast(DType::I64));
            check!(Tensor::<B, 2, Bool>::from_data([[true, false]], &device));
            check!(Tensor::<B, 2>::empty([0, 3], &device));
        }

        #[test]
        #[should_panic(expected = "Empty Like")]
        fn empty_like_rejects_quantized_dtype() {
            let device = Default::default();
            Tensor::<B, 1>::from_floats([-1., 2.], &device)
                .quantize_dynamic(&QuantScheme::default())
                .empty_like();
        }

        #[test]
        fn six_float_scalar_comparisons() {
            let device = Default::default();
            let tensor = Tensor::<B, 1>::from_floats([-2., 0., 3.], &device);
            for (output, expected) in [
                (tensor.clone().equal_scalar(0.), [false, true, false]),
                (tensor.clone().not_equal_scalar(0.), [true, false, true]),
                (tensor.clone().greater_scalar(0.), [false, false, true]),
                (tensor.clone().greater_equal_scalar(0.), [false, true, true]),
                (tensor.clone().lower_scalar(0.), [true, false, false]),
                (tensor.lower_equal_scalar(0.), [true, true, false]),
            ] {
                assert_eq!(output.into_data().try_into_vec_as::<bool>().unwrap(), expected);
            }
        }

        #[test]
        fn integer_scalar_comparisons_preserve_i64_precision() {
            let device = Default::default();
            let middle = (1i64 << 53) + 1;
            let tensor = Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![middle - 1, middle, middle + 1], [3]),
                (&device, DType::I64),
            );
            for (output, expected) in [
                (tensor.clone().equal_scalar(middle), [false, true, false]),
                (tensor.clone().not_equal_scalar(middle), [true, false, true]),
                (tensor.clone().greater_scalar(middle), [false, false, true]),
                (
                    tensor.clone().greater_equal_scalar(middle),
                    [false, true, true],
                ),
                (tensor.clone().lower_scalar(middle), [true, false, false]),
                (tensor.lower_equal_scalar(middle), [true, true, false]),
            ] {
                assert_eq!(output.into_data().try_into_vec_as::<bool>().unwrap(), expected);
            }
        }

        #[test]
        fn scalar_comparisons_preserve_bool_nan_and_signed_zero() {
            let device = Default::default();
            let boolean = Tensor::<B, 1, Bool>::from_data([true, false], &device);
            assert_eq!(
                boolean
                    .clone()
                    .equal_scalar(true)
                    .into_data()
                    .try_into_vec_as::<bool>()
                    .unwrap(),
                [true, false]
            );
            assert_eq!(
                boolean
                    .not_equal_scalar(true)
                    .into_data()
                    .try_into_vec_as::<bool>()
                    .unwrap(),
                [false, true]
            );
            let floating = Tensor::<B, 1>::from_floats([f32::NAN, -0., 0.], &device);
            assert_eq!(
                floating
                    .clone()
                    .equal_scalar(0.)
                    .into_data()
                    .try_into_vec_as::<bool>()
                    .unwrap(),
                [false, true, true]
            );
            assert_eq!(
                floating
                    .not_equal_scalar(0.)
                    .into_data()
                    .try_into_vec_as::<bool>()
                    .unwrap(),
                [true, false, false]
            );
        }

        #[test]
        fn quantized_ownership_tracks_clones() {
            let device = Default::default();
            let tensor = Tensor::<B, 1>::from_floats([-1., 2.], &device)
                .quantize_dynamic(&QuantScheme::default());
            B::sync(&device).unwrap();
            assert!(tensor.can_mut());
            let alias = tensor.clone();
            assert!(!tensor.can_mut());
            assert!(!alias.can_mut());
            drop(alias);
            assert!(tensor.can_mut());
        }
    };
}
