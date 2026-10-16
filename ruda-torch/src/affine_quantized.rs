use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn clamp_integer(value: i64, minimum: i64, maximum: i64) -> i64 {
    let mut result = value;
    if result < minimum { result = minimum; }
    if result > maximum { result = maximum; }
    result
}

#[ruda(launch)]
fn quantize_scalar<Q: Numeric + RudaElement>(
    input: &Tensor<f32>, output: &mut Tensor<Q>, scale: f64, zero: i64, minimum: i64, maximum: i64,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let x = f64::cast_from(input[address(input, p)]);
        let q = i64::cast_from((x / scale).round() + f64::cast_from(zero));
        output[address(output, p)] = Q::cast_from(clamp_integer(q, minimum, maximum));
    }
}

#[ruda(launch)]
fn dequantize_scalar<Q: Numeric + RudaElement>(
    input: &Tensor<Q>, output: &mut Tensor<f32>, scale: f64, zero: i64,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let difference = f32::cast_from(input[address(input, p)]) - f32::cast_from(zero);
        output[address(output, p)] = f32::cast_from(f64::cast_from(difference) * scale);
    }
}

#[ruda(launch)]
fn quantize_channel<Q: Numeric + RudaElement, S: Float + RudaElement, Z: Numeric + RudaElement>(
    input: &Tensor<f32>, output: &mut Tensor<Q>, scales: &Tensor<S>, zeros: &Tensor<Z>,
    minimum: i64, maximum: i64, #[comptime] inner: usize, #[comptime] channels: usize,
    #[comptime] floating_zero: bool,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let channel = p / inner % channels;
        let scale = scales[address(scales, channel)];
        let zero = zeros[address(zeros, channel)];
        let x = input[address(input, p)];
        let mut q = 0i64;
        if comptime!(floating_zero) {
            let inv = 1.0f32 / f32::cast_from(scale);
            q = i64::cast_from((x * inv + f32::cast_from(zero)).round());
        } else {
            q = i64::cast_from((f64::cast_from(x) / f64::cast_from(scale)).round() + f64::cast_from(zero));
        }
        output[address(output, p)] = Q::cast_from(clamp_integer(q, minimum, maximum));
    }
}

#[ruda(launch)]
fn dequantize_channel<Q: Numeric + RudaElement, S: Float + RudaElement, Z: Numeric + RudaElement>(
    input: &Tensor<Q>, output: &mut Tensor<f32>, scales: &Tensor<S>, zeros: &Tensor<Z>,
    #[comptime] inner: usize, #[comptime] channels: usize, #[comptime] floating_zero: bool,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let channel = p / inner % channels;
        let scale = scales[address(scales, channel)];
        let zero = zeros[address(zeros, channel)];
        let q = input[address(input, p)];
        let mut value = 0.0f32;
        if comptime!(floating_zero) {
            value = (f32::cast_from(q) - f32::cast_from(zero)) * f32::cast_from(scale);
        } else {
            let difference = f32::cast_from(i64::cast_from(q) - i64::cast_from(zero));
            value = f32::cast_from(f64::cast_from(difference) * f64::cast_from(scale));
        }
        output[address(output, p)] = value;
    }
}

fn grid(n: usize) -> RudaCount {
    RudaCount::Static(u32::try_from(n.div_ceil(256)).expect("quantization launch overflow"), 1, 1)
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64], values: &[f64]) {
    assert!((549..=551).contains(&op));
    assert_eq!(ints.len(), 2); assert!(ints[1] == 0 || ints[1] == 1);
    assert_eq!(xs.len(), if op == 549 { 2 } else { 4 });
    assert_eq!(xs[0].dtype, 0); assert_eq!(xs[0].shape, xs[1].shape);
    let (minimum, maximum): (i64, i64) = match xs[1].dtype {
        5 => (i32::MIN as i64, i32::MAX as i64), 7 => (-128, 127), 8 => (0, 255),
        _ => panic!("quantized storage must be int8, uint8 or int32"),
    };
    let c = client(); let block = RudaDim::new_1d(256);
    let reverse = ints[1] != 0;
    if op == 549 {
        assert_eq!(values.len(), 1); assert!(ints[0] >= minimum && ints[0] <= maximum);
        if xs[0].len == 0 { return; }
        macro_rules! run { ($q:ty) => { unsafe {
            if reverse {
                dequantize_scalar::launch::<$q,CudaRuntime>(&c,grid(xs[0].len),block,
                    xs[1].arg(),xs[0].arg(),values[0],ints[0]);
            } else {
                quantize_scalar::launch::<$q,CudaRuntime>(&c,grid(xs[0].len),block,
                    xs[0].arg(),xs[1].arg(),values[0],ints[0],minimum,maximum);
            }
        } }; }
        match xs[1].dtype { 5=>run!(i32),7=>run!(i8),8=>run!(u8),_=>unreachable!() }
    } else {
        assert!(values.is_empty());
        let axis = usize::try_from(ints[0]).expect("negative quantization channel axis");
        assert!(axis < xs[0].shape.len());
        let channels = xs[0].shape[axis];
        let inner = xs[0].shape[axis + 1..].iter().product::<usize>();
        assert_eq!(xs[2].len, channels); assert_eq!(xs[3].len, channels);
        assert_eq!(xs[2].dtype, if op == 550 { 9 } else { 0 });
        assert_eq!(xs[3].dtype, if op == 550 { 4 } else { 0 });
        if xs[0].len == 0 { return; }
        macro_rules! run { ($q:ty,$s:ty,$z:ty) => { unsafe {
            if reverse {
                dequantize_channel::launch::<$q,$s,$z,CudaRuntime>(&c,grid(xs[0].len),block,
                    xs[1].arg(),xs[0].arg(),xs[2].arg(),xs[3].arg(),inner,channels,op==551);
            } else {
                quantize_channel::launch::<$q,$s,$z,CudaRuntime>(&c,grid(xs[0].len),block,
                    xs[0].arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),minimum,maximum,inner,channels,op==551);
            }
        } }; }
        macro_rules! storage { ($s:ty,$z:ty) => { match xs[1].dtype {
            5=>run!(i32,$s,$z),7=>run!(i8,$s,$z),8=>run!(u8,$s,$z),_=>unreachable!()
        } }; }
        if op == 550 { storage!(f64,i64); } else { storage!(f32,f32); }
    }
}
