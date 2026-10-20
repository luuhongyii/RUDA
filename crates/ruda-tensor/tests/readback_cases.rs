macro_rules! readback_cases {
    ($backend:ty) => {
        use ruda_tensor::api::{Bool, DType, DataError, Int, Tensor, TensorData, TensorReadError};
        type B = $backend;

        #[test]
        fn resident_readback_access_policy() {
            let device = Default::default();
            let tensor = Tensor::<B, 1>::from_floats([1.0, 2.0], &device);
            let mut data = tensor.try_to_data().unwrap();
            assert_eq!(data.bytes.read(ruda_core::bytes::Reader::new().no_copy()).unwrap().len(), 8);
            data.bytes.write(ruda_core::bytes::Writer::new()).unwrap()[..4].copy_from_slice(&42.0f32.to_ne_bytes());
            assert_eq!(data.try_into_vec::<f32>().unwrap(), [42.0, 2.0]);
            assert_eq!(tensor.try_to_vec_as::<f32>().unwrap(), [1.0, 2.0]);
        }

        #[test]
        fn borrowed_read_preserves_tensor() {
            let device = Default::default();
            let tensor = Tensor::<B, 1>::from_floats([-2.0, 0.0, 3.5], &device);
            assert_eq!(tensor.try_to_data().unwrap().to_vec::<f32>().unwrap(), [-2.0, 0.0, 3.5]);
            assert_eq!(tensor.try_to_vec_as::<i32>().unwrap(), [-2, 0, 3]);
            assert_eq!(tensor.try_to_vec_as::<f64>().unwrap(), [-2.0, 0.0, 3.5]);
            assert_eq!(tensor.into_data().to_vec::<f32>().unwrap(), [-2.0, 0.0, 3.5]);
        }

        #[test]
        fn borrowed_and_owned_typed_data() {
            let device = Default::default();
            let tensor = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
            for data in [tensor.to_data_as::<f64>(), tensor.try_to_data_as::<f64>().unwrap(),
                         tensor.clone().into_data_as::<f64>(), tensor.try_into_data_as::<f64>().unwrap()] {
                assert_eq!(data.dtype, DType::F64);
                assert_eq!(data.shape.as_slice(), [2, 2]);
                assert_eq!(data.try_into_vec::<f64>().unwrap(), [1.0, 2.0, 3.0, 4.0]);
            }
        }

        #[test]
        fn borrowed_and_owned_runtime_dtype() {
            let device = Default::default();
            let tensor = Tensor::<B, 1>::from_floats([1.5, -2.5], &device);
            for data in [tensor.to_data_dtype(DType::I32), tensor.try_to_data_dtype(DType::I32).unwrap(),
                         tensor.clone().into_data_dtype(DType::I32), tensor.try_into_data_dtype(DType::I32).unwrap()] {
                assert_eq!(data.dtype, DType::I32);
                assert_eq!(data.try_into_vec::<i32>().unwrap(), [1, -2]);
            }
        }

        #[test]
        fn owned_vector_and_noncontiguous_layout() {
            let device = Default::default();
            let tensor = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device).swap_dims(0, 1);
            assert_eq!(tensor.try_into_vec_as::<i64>().unwrap(), [1, 4, 2, 5, 3, 6]);
        }

        #[test]
        fn boolean_and_integer_readback() {
            let device = Default::default();
            let boolean = Tensor::<B, 1, Bool>::from_data([true, false, true], &device);
            assert_eq!(boolean.try_to_vec_as::<bool>().unwrap(), [true, false, true]);
            assert_eq!(boolean.try_into_vec_as::<u32>().unwrap(), [1, 0, 1]);
            let integer = Tensor::<B, 1, Int>::from_data([-3, 0, 7], &device);
            assert_eq!(integer.try_to_vec_as::<f64>().unwrap(), [-3.0, 0.0, 7.0]);
            assert_eq!(integer.try_into_vec_as::<bool>().unwrap(), [true, false, true]);
        }

        #[test]
        fn empty_readback_keeps_shape() {
            let device = Default::default();
            let tensor = Tensor::<B, 2>::empty([0, 3], &device);
            let data = tensor.try_to_data_as::<f64>().unwrap();
            assert_eq!(data.shape.as_slice(), [0, 3]);
            assert!(data.try_into_vec::<f64>().unwrap().is_empty());
            assert!(tensor.try_into_vec_as::<i32>().unwrap().is_empty());
        }

        #[test]
        fn typed_scalar_sync_and_async() {
            let device = Default::default();
            let tensor = Tensor::<B, 2, Int>::from_data([[7]], &device);
            assert_eq!(tensor.clone().try_into_scalar_as::<f64>().unwrap(), 7.0);
            assert_eq!(ruda_tensor::api::try_read_sync(tensor.try_into_scalar_async::<i64>()).unwrap().unwrap(), 7);
        }

        #[test]
        fn invalid_scalar_shapes_return_errors() {
            let device = Default::default();
            for values in [vec![], vec![1.0, 2.0]] {
                let expected = values.len();
                let tensor = Tensor::<B, 1>::from_data(TensorData::new(values, [expected]), &device);
                assert!(matches!(tensor.clone().try_into_scalar_as::<f32>(),
                    Err(TensorReadError::InvalidShape { expected: 1, actual }) if actual == expected));
                assert!(matches!(ruda_tensor::api::try_read_sync(tensor.try_into_scalar_async::<f32>()).unwrap(),
                    Err(TensorReadError::InvalidShape { expected: 1, actual }) if actual == expected));
            }
        }

        #[test]
        fn unsupported_quantized_target_is_data_error() {
            let device = Default::default();
            let tensor = Tensor::<B, 1>::from_floats([1.0], &device);
            let target = DType::QFloat(Default::default());
            assert!(matches!(tensor.try_to_data_dtype(target),
                Err(TensorReadError::Data(DataError::UnsupportedConversion { from: DType::F32, to })) if to == target));
            assert!(matches!(tensor.try_into_data_dtype(target),
                Err(TensorReadError::Data(DataError::UnsupportedConversion { from: DType::F32, to })) if to == target));
        }
    };
}
