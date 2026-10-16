use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn masked_fill<Q: Numeric + RudaElement>(
    mask: &Tensor<u8>, output: &mut Tensor<Q>, value: f32, scale: f64,
    zero: i64, minimum: i64, maximum: i64,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        if mask[address(mask, p)] != 0u8 {
            let inverse = 1.0f32 / f32::cast_from(scale);
            let rounded = (f32::cast_from(zero) + value * inverse).round();
            let mut q = f64::cast_from(rounded);
            if q < f64::cast_from(minimum) { q = f64::cast_from(minimum); }
            if q > f64::cast_from(maximum) { q = f64::cast_from(maximum); }
            output[address(output, p)] = Q::cast_from(q);
        }
    }
}

#[ruda(launch)]
fn index_roundtrip<Q: Numeric + RudaElement>(
    input: &Tensor<Q>, output: &mut Tensor<Q>, scale: f64, zero: i64, minimum: i64, maximum: i64,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let difference = f32::cast_from(input[address(input, p)]) - f32::cast_from(zero);
        let real = f32::cast_from(f64::cast_from(difference) * scale);
        let mut q = i64::cast_from((f64::cast_from(real) / scale).round() + f64::cast_from(zero));
        if q < minimum { q = minimum; }
        if q > maximum { q = maximum; }
        output[address(output, p)] = Q::cast_from(q);
    }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64], values: &[f64]) {
    assert!(op == 576 || op == 577); assert_eq!(xs.len(), 2); assert_eq!(ints.len(), 1);
    assert_eq!(values.len(), if op == 576 { 2 } else { 1 });
    let (input, output) = (&xs[0], &xs[1]); assert_eq!(input.shape, output.shape);
    assert_eq!(input.dtype, if op == 576 { 3 } else { output.dtype });
    let (minimum, maximum): (i64, i64) = match output.dtype {
        5 => (i32::MIN as i64, i32::MAX as i64), 7 => (-128,127), 8 => (0,255),
        _ => panic!("unsupported quantized index storage"),
    };
    assert!((minimum..=maximum).contains(&ints[0]));
    if output.len == 0 { return; }
    let c = client(); let block = RudaDim::new_1d(256);
    let grid = RudaCount::Static(u32::try_from(output.len.div_ceil(256)).expect("quantized index launch overflow"),1,1);
    macro_rules! run { ($q:ty) => { unsafe {
        if op == 576 {
            masked_fill::launch::<$q,CudaRuntime>(&c,grid,block,input.arg(),output.arg(),
                values[1] as f32,values[0],ints[0],minimum,maximum);
        } else {
            index_roundtrip::launch::<$q,CudaRuntime>(&c,grid,block,input.arg(),output.arg(),
                values[0],ints[0],minimum,maximum);
        }
    } }; }
    match output.dtype { 5=>run!(i32),7=>run!(i8),8=>run!(u8),_=>unreachable!() }
}
