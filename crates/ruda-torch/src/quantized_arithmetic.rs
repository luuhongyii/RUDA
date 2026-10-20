use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn rounded_integer(value: f32, zero: i64, minimum: i64, maximum: i64) -> i64 {
    let mut q = f64::cast_from(value.round()) + f64::cast_from(zero);
    if q < f64::cast_from(minimum) { q = f64::cast_from(minimum); }
    if q > f64::cast_from(maximum) { q = f64::cast_from(maximum); }
    i64::cast_from(q)
}

#[ruda(launch)]
fn binary<Q: Numeric + RudaElement>(
    a: &Tensor<Q>, b: &Tensor<Q>, output: &mut Tensor<Q>,
    a_scale: f32, b_scale: f32, output_scale: f32,
    a_zero: i64, b_zero: i64, output_zero: i64, minimum: i64, maximum: i64,
    #[comptime] multiply: bool, #[comptime] relu: bool,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let av = a[address(a, p)]; let bv = b[address(b, p)];
        let inv_scale = 1.0f32 / output_scale;
        let mut scaled = 0.0f32;
        if comptime!(multiply) {
            let ai = u32::cast_from(i32::cast_from(av)) - u32::cast_from(i32::cast_from(a_zero));
            let bi = u32::cast_from(i32::cast_from(bv)) - u32::cast_from(i32::cast_from(b_zero));
            let product = i32::cast_from(ai * bi);
            let multiplier = a_scale * b_scale * inv_scale;
            scaled = f32::cast_from(f64::cast_from(product) * f64::cast_from(multiplier));
        } else {
            let da = a_scale * f32::cast_from(i64::cast_from(av) - a_zero);
            let db = b_scale * f32::cast_from(i64::cast_from(bv) - b_zero);
            let mut value = da + db;
            if comptime!(relu) { if value < 0.0f32 { value = 0.0f32; } }
            scaled = value * inv_scale;
        }
        let mut q = rounded_integer(scaled, output_zero, minimum, maximum);
        if comptime!(relu) { if q < output_zero { q = output_zero; } }
        output[address(output, p)] = Q::cast_from(q);
    }
}

#[ruda(launch)]
fn scalar<Q: Numeric + RudaElement>(
    input: &Tensor<Q>, output: &mut Tensor<Q>, input_scale: f32, output_scale: f32,
    input_zero: i64, output_zero: i64, offset: i32, minimum: i64, maximum: i64,
    #[comptime] mode: u32, #[comptime] relu: bool,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let value = input[address(input, p)];
        let mut q = i64::cast_from(value);
        if comptime!(mode == 1) {
            let difference = u32::cast_from(i32::cast_from(value)) - u32::cast_from(i32::cast_from(input_zero));
            let sum = i32::cast_from(difference + u32::cast_from(offset));
            let multiplier = input_scale * (1.0f32 / output_scale);
            let scaled = f32::cast_from(f64::cast_from(sum) * f64::cast_from(multiplier));
            q = rounded_integer(scaled, output_zero, minimum, maximum);
        } else if comptime!(mode == 2) {
            q = 0i64;
        } else if comptime!(mode == 3) {
            q = maximum + minimum - q;
        }
        if comptime!(relu) { if q < output_zero { q = output_zero; } }
        output[address(output, p)] = Q::cast_from(q);
    }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64], values: &[f64]) {
    assert!(op == 554 || op == 555);
    assert_eq!(xs.len(), if op == 554 { 3 } else { 2 });
    assert_eq!(ints.len(), 5); assert_eq!(values.len(), if op == 554 { 3 } else { 2 });
    assert!((0..=1).contains(&ints[1]));
    let output = xs.last().unwrap();
    for x in xs { assert_eq!(x.shape, output.shape); assert_eq!(x.dtype, output.dtype); }
    let (minimum, maximum): (i64, i64) = match output.dtype {
        5 => (i32::MIN as i64, i32::MAX as i64), 7 => (-128, 127), 8 => (0, 255),
        _ => panic!("quantized arithmetic storage must be int8, uint8 or int32"),
    };
    assert!((minimum..=maximum).contains(&ints[2]));
    assert!((minimum..=maximum).contains(&ints[3]));
    if op == 554 {
        assert!((0..=1).contains(&ints[0])); assert!((minimum..=maximum).contains(&ints[4]));
    } else {
        assert!((0..=3).contains(&ints[0])); assert!(i32::try_from(ints[4]).is_ok());
    }
    if output.len == 0 { return; }
    let c = client(); let block = RudaDim::new_1d(256);
    let grid = RudaCount::Static(u32::try_from(output.len.div_ceil(256)).expect("quantized arithmetic launch overflow"), 1, 1);
    macro_rules! run { ($q:ty) => { unsafe {
        if op == 554 {
            binary::launch::<$q,CudaRuntime>(&c,grid,block,xs[0].arg(),xs[1].arg(),output.arg(),
                values[0] as f32,values[1] as f32,values[2] as f32,
                ints[2],ints[3],ints[4],minimum,maximum,ints[0]!=0,ints[1]!=0);
        } else {
            scalar::launch::<$q,CudaRuntime>(&c,grid,block,xs[0].arg(),output.arg(),
                values[0] as f32,values[1] as f32,ints[2],ints[3],ints[4] as i32,
                minimum,maximum,ints[0] as u32,ints[1]!=0);
        }
    } }; }
    match output.dtype { 5=>run!(i32),7=>run!(i8),8=>run!(u8),_=>unreachable!() }
}
