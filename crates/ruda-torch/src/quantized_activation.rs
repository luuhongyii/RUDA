use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn quantized_value(value: f32, scale: f32, zero: i64, minimum: i64, maximum: i64) -> i64 {
    let inv_scale = 1.0f32 / scale;
    let mut q = f64::cast_from((value * inv_scale).round()) + f64::cast_from(zero);
    if q < f64::cast_from(minimum) { q = f64::cast_from(minimum); }
    if q > f64::cast_from(maximum) { q = f64::cast_from(maximum); }
    i64::cast_from(q)
}

#[ruda(launch)]
fn bounded<Q: Numeric + RudaElement>(
    input: &Tensor<Q>, output: &mut Tensor<Q>, scale: f32, zero: i64,
    lower: f32, upper: f32, minimum: i64, maximum: i64,
    #[comptime] relu: bool, #[comptime] has_lower: bool, #[comptime] has_upper: bool,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let mut value = i64::cast_from(input[address(input, p)]);
        if comptime!(relu) {
            if value < zero { value = zero; }
        } else {
            if comptime!(has_lower) {
                let bound = quantized_value(lower, scale, zero, minimum, maximum);
                if value < bound { value = bound; }
            }
            if comptime!(has_upper) {
                let bound = quantized_value(upper, scale, zero, minimum, maximum);
                if value > bound { value = bound; }
            }
        }
        output[address(output, p)] = Q::cast_from(value);
    }
}

#[ruda(launch)]
fn nonlinear<Q: Numeric + RudaElement>(
    input: &Tensor<Q>, output: &mut Tensor<Q>, input_scale: f32, input_zero: i64,
    output_scale: f32, output_zero: i64, p0: f32, p1: f32, p2: f32,
    minimum: i64, maximum: i64, #[comptime] mode: u32,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let difference = i64::cast_from(input[address(input, p)]) - input_zero;
        let x = input_scale * f32::cast_from(difference);
        let mut y = x;
        if comptime!(mode == 0) {
            if !(x > 0.0f32) { y = x * p0; }
        } else if comptime!(mode == 1) {
            y = 1.0f32 / (1.0f32 + (-x).exp());
        } else if comptime!(mode == 2) {
            y = x.tanh();
        } else if comptime!(mode == 3 || mode == 5) {
            let mut gate = x + 3.0f32;
            if gate < 0.0f32 { gate = 0.0f32; }
            if gate > 6.0f32 { gate = 6.0f32; }
            y = gate / 6.0f32;
            if comptime!(mode == 5) { y = (x * gate) / 6.0f32; }
        } else if comptime!(mode == 4) {
            if !(x > p0) { y = p1; }
        } else if comptime!(mode == 6) {
            y = x * p1;
            if !(x >= 0.0f32) {
                y = f32::cast_from(crate::elementary::expm1(f64::cast_from(x * p2))) * p0 * p1;
            }
        } else if comptime!(mode == 7) {
            let alpha = f32::cast_from(0.707106781186547524400844362104849f64);
            y = x * 0.5f32 * (1.0f32 + (x * alpha).erf());
        } else if comptime!(mode == 8) {
            let beta = f32::cast_from(0.797884560802865355879892119868764f64);
            let cube = x * x * x;
            y = 0.5f32 * x * (1.0f32 + (beta * (x + 0.044715f32 * cube)).tanh());
        }
        output[address(output, p)] = Q::cast_from(quantized_value(y, output_scale, output_zero, minimum, maximum));
    }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64], values: &[f64]) {
    assert!(op == 552 || op == 553); assert_eq!(xs.len(), 2);
    assert_eq!(xs[0].shape, xs[1].shape); assert_eq!(xs[0].dtype, xs[1].dtype);
    let (minimum, maximum): (i64, i64) = match xs[0].dtype {
        5 => (i32::MIN as i64, i32::MAX as i64), 7 => (-128, 127), 8 => (0, 255),
        _ => panic!("quantized activation storage must be int8, uint8 or int32"),
    };
    assert_eq!(ints.len(), if op == 552 { 4 } else { 3 });
    assert_eq!(values.len(), if op == 552 { 3 } else { 5 });
    assert!((minimum..=maximum).contains(&ints[1]));
    if op == 552 {
        assert!(ints[0] == 0 || ints[0] == 1);
        assert!((0..=1).contains(&ints[2]) && (0..=1).contains(&ints[3]));
        assert!(ints[0] == 0 || ints[2] != 0 || ints[3] != 0);
    } else {
        assert!((0..=8).contains(&ints[0])); assert!((minimum..=maximum).contains(&ints[2]));
    }
    if xs[0].len == 0 { return; }
    let c = client(); let block = RudaDim::new_1d(256);
    let grid = RudaCount::Static(u32::try_from(xs[0].len.div_ceil(256)).expect("quantized activation launch overflow"), 1, 1);
    macro_rules! run { ($q:ty) => { unsafe {
        if op == 552 {
            bounded::launch::<$q,CudaRuntime>(&c,grid,block,xs[0].arg(),xs[1].arg(),
                values[0] as f32,ints[1],values[1] as f32,values[2] as f32,minimum,maximum,
                ints[0]==0,ints[2]!=0,ints[3]!=0);
        } else {
            nonlinear::launch::<$q,CudaRuntime>(&c,grid,block,xs[0].arg(),xs[1].arg(),
                values[0] as f32,ints[1],values[1] as f32,ints[2],
                values[2] as f32,values[3] as f32,values[4] as f32,minimum,maximum,ints[0] as u32);
        }
    } }; }
    match xs[0].dtype { 5=>run!(i32),7=>run!(i8),8=>run!(u8),_=>unreachable!() }
}
