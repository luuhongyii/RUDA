include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tests/ownership_cases.rs"
));
ownership_cases!(crate::Router<(ruda_tensor_host::Host, ruda_tensor_host::Host)>);

#[test]
fn integer_casts_preserve_values_dtypes_shapes_and_aliases() {
    let device = Default::default();
    let original = Tensor::<B, 2, Int>::from_data([[0, 1], [42, 127]], &device);
    for dtype in [
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
    ] {
        let output = original.clone().transpose().cast(dtype);
        assert_eq!(output.dtype(), dtype);
        assert_eq!(output.dims(), [2, 2]);
        assert_eq!(output.try_into_vec_as::<i64>().unwrap(), [0, 42, 1, 127]);
    }
    assert_eq!(
        original.into_data().to_vec::<i32>().unwrap(),
        [0, 1, 42, 127]
    );
    for dtype in [DType::I8, DType::I16, DType::I32, DType::I64] {
        let input = Tensor::<B, 1, Int>::from_data([-128, -1, 0, 127], &device);
        assert_eq!(
            input.cast(dtype).try_into_vec_as::<i64>().unwrap(),
            [-128, -1, 0, 127]
        );
    }
    let large = [-(1i64 << 53) - 1, (1i64 << 53) + 1];
    let input =
        Tensor::<B, 1, Int>::from_data(TensorData::new(large.to_vec(), [2]), (&device, DType::I64));
    let alias = input.clone();
    assert_eq!(
        input.cast(DType::I64).into_data().to_vec::<i64>().unwrap(),
        large
    );
    assert_eq!(alias.into_data().to_vec::<i64>().unwrap(), large);
}
