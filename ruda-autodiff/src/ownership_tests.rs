include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../ruda-tensor/tests/ownership_cases.rs"
));
ownership_cases!(crate::Autodiff<ruda_tensor_host::Host>);

#[test]
fn extract_preserves_autodiff_lineage() {
    let device = Default::default();
    let mut tensor = Tensor::<B, 1>::from_floats([2., -3.], &device).require_grad();
    assert!(tensor.can_mut());
    let parent = tensor.clone();
    assert!(!tensor.can_mut());
    let output = tensor.extract().square();
    assert_eq!(tensor.dims(), [0]);
    let gradients = output.backward();
    assert_eq!(
        parent
            .grad(&gradients)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
        [4., -6.]
    );
}
