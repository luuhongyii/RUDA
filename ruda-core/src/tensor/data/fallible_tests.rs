use super::{DataError, TensorData};
use crate::{
    bytes::{AccessError, Bytes},
    tensor::{BoolStore, DType, QuantScheme},
};
use alloc::vec;

#[test]
fn typed_vectors_and_all_numeric_casts() {
    let types = [
        DType::F64,
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::I64,
        DType::I32,
        DType::I16,
        DType::I8,
        DType::U64,
        DType::U32,
        DType::U16,
        DType::U8,
    ];
    for source in types {
        for target in types {
            let data = TensorData::from([0i32, 1, 7]).try_cast(source).unwrap();
            let result = data.try_cast(target).unwrap();
            assert_eq!(result.dtype, target);
            assert_eq!(result.try_to_vec_as::<i32>().unwrap(), [0, 1, 7]);
            assert_eq!(result.try_into_vec_as::<f64>().unwrap(), [0.0, 1.0, 7.0]);
        }
    }
}

#[test]
fn bool_storage_is_normalized_for_both_conversion_paths() {
    for dtype in [
        DType::Bool(BoolStore::Native),
        DType::Bool(BoolStore::U8),
        DType::Bool(BoolStore::U32),
    ] {
        for data in [
            TensorData::from([-2i32, 0, 4]),
            TensorData::from([-2i8, 0, 4]),
            TensorData::from([-0.5f64, 0.0, 4.5]),
        ] {
            let result = data.try_cast(dtype).unwrap();
            assert_eq!(result.dtype, dtype);
            assert_eq!(result.try_to_vec_as::<bool>().unwrap(), [true, false, true]);
            assert_eq!(result.try_into_vec_as::<u32>().unwrap(), [1, 0, 1]);
        }
    }
}

#[test]
fn invalid_storage_and_counts_match_cast_contract() {
    for target in [DType::I32, DType::F64] {
        let data = TensorData::from_bytes(Bytes::from_bytes_vec(vec![0; 3]), [1], DType::F32);
        assert!(matches!(
            data.try_cast(target),
            Err(DataError::InvalidRepresentation(_))
        ));
        let data = TensorData::new(vec![1.0f32, 2.0], [2]);
        let data = TensorData::from_bytes(data.bytes, [1], DType::F32);
        assert!(matches!(
            data.try_cast(target),
            Err(DataError::ElementCountMismatch {
                expected: 1,
                actual: 2
            })
        ));
    }
    let data = TensorData::from_bytes(
        Bytes::from_bytes_vec(vec![2]),
        [1],
        DType::Bool(BoolStore::Native),
    );
    assert!(matches!(
        data.try_to_vec::<bool>(),
        Err(DataError::InvalidRepresentation(_))
    ));
    assert!(matches!(
        data.try_into_vec::<bool>(),
        Err(DataError::InvalidRepresentation(_))
    ));
}

#[test]
fn dtype_mismatch_is_not_an_implicit_cast() {
    let data = TensorData::from([1.0f32]);
    assert!(matches!(
        data.try_to_vec::<i32>(),
        Err(DataError::DTypeMismatch {
            expected: DType::I32,
            actual: DType::F32
        })
    ));
    assert!(matches!(
        data.try_into_vec::<i32>(),
        Err(DataError::DTypeMismatch {
            expected: DType::I32,
            actual: DType::F32
        })
    ));
}

#[test]
fn unsupported_quantized_casts_return_errors() {
    let scheme = QuantScheme::default();
    let target = DType::QFloat(scheme);
    assert!(
        matches!(TensorData::from([1.0f32]).try_cast(target), Err(DataError::UnsupportedConversion { from: DType::F32, to }) if to == target)
    );
    let quantized = TensorData::quantized(vec![0i8], [1], scheme, &[1.0]);
    assert!(
        matches!(quantized.try_cast(DType::F32), Err(DataError::UnsupportedConversion { from, to: DType::F32 }) if from == target)
    );
}

#[test]
fn matching_owned_storage_is_reused() {
    let data = TensorData::from([1.0f32, 2.0]);
    let pointer = data.bytes.as_ptr();
    let values = data.try_into_vec::<f32>().unwrap();
    assert_eq!(values.as_ptr().cast::<u8>(), pointer);
    assert_eq!(values, [1.0, 2.0]);
}

#[test]
fn empty_data_conversion_preserves_shape() {
    let data = TensorData::new(alloc::vec::Vec::<i32>::new(), [0, 3]);
    let data = data.try_cast_as::<f64>().unwrap();
    assert_eq!(data.shape.as_slice(), [0, 3]);
    assert!(data.try_into_vec::<f64>().unwrap().is_empty());
}

#[cfg(feature = "std")]
#[test]
fn real_file_errors_are_returned_and_retryable() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("data");
    let data = TensorData::from_bytes(Bytes::from_file(&path, 8, 0), [2], DType::I32);
    assert!(matches!(
        data.try_to_vec::<i32>(),
        Err(DataError::StorageAccess(AccessError::Read(_)))
    ));
    assert!(matches!(
        data.clone().try_cast(DType::F32),
        Err(DataError::StorageAccess(AccessError::Read(_)))
    ));
    assert!(matches!(
        data.clone().try_cast(DType::F64),
        Err(DataError::StorageAccess(AccessError::Read(_)))
    ));
    std::fs::write(&path, [1i32.to_ne_bytes(), 2i32.to_ne_bytes()].concat()).unwrap();
    assert_eq!(data.try_to_vec::<i32>().unwrap(), [1, 2]);
    assert_eq!(data.try_into_vec_as::<f64>().unwrap(), [1.0, 2.0]);
}
