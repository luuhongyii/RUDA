#[path = "../src/compiler_backend.rs"]
mod compiler_backend;

#[test]
fn scalar_copy_layout_preserves_rank_validation() {
    use ruda_driver_cuda::CudaRuntime;
    use ruda_kernel::dsl::{Runtime, zspace::{Shape, Strides}};

    assert!(CudaRuntime::can_read_tensor(&Shape::from([0usize; 0]), &Strides::from([0usize; 0])));
    assert!(CudaRuntime::can_read_tensor(&Shape::from([1]), &Strides::from([1])));
    assert!(!CudaRuntime::can_read_tensor(&Shape::from([2, 2]), &Strides::from([1, 2])));
    assert!(std::panic::catch_unwind(|| {
        CudaRuntime::can_read_tensor(&Shape::from([0usize; 0]), &Strides::from([1]))
    }).is_err());
}

#[test]
fn cuda_nonfinite_constants_preserve_float_bits() {
    use ruda_compiler::cpp::{cuda::{CudaDialect, mma::CudaWmmaCompiler}, shared::{Elem, Item, Variable}};
    use ruda_core::ir::ConstantValue;
    type D = CudaDialect<CudaWmmaCompiler>;

    for bits in [f64::INFINITY.to_bits(), f64::NEG_INFINITY.to_bits(),
        0x7ff8_0000_0000_0042, 0xfff8_0000_0000_0042] {
        for (elem, name) in [(Elem::<D>::F32, "float"), (Elem::<D>::F64, "double")] {
            let value = Variable::Constant(ConstantValue::Float(f64::from_bits(bits)), Item::scalar(elem, false));
            assert_eq!(value.to_string(), format!("{name}(__longlong_as_double({}LL))", bits as i64));
        }
    }
}

#[test]
fn cuda_finite_constants_keep_existing_format() {
    use ruda_compiler::cpp::{cuda::{CudaDialect, mma::CudaWmmaCompiler}, shared::{Elem, Item, Variable}};
    use ruda_core::ir::ConstantValue;
    type D = CudaDialect<CudaWmmaCompiler>;
    for number in [-0.0, 0.0, 1.25, f64::MIN_POSITIVE, f64::MAX] {
        let value = Variable::Constant(ConstantValue::Float(number), Item::<D>::scalar(Elem::F64, false));
        assert_eq!(value.to_string(), format!("double({number:?})"));
    }
}
