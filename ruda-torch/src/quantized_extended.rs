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

#[ruda]
fn affine_value<Q: Numeric>(value: Q, scale: f64, zero: i64) -> f32 {
    let difference = f32::cast_from(value) - f32::cast_from(zero);
    f32::cast_from(f64::cast_from(difference) * scale)
}

#[ruda]
fn affine_integer(value: f32, scale: f64, zero: i64, minimum: i64, maximum: i64) -> i64 {
    let q = i64::cast_from((f64::cast_from(value) / scale).round() + f64::cast_from(zero));
    clamp_integer(q, minimum, maximum)
}

#[ruda(launch)]
fn concatenate_segment<Q: Numeric + RudaElement>(
    input: &Tensor<Q>, output: &mut Tensor<Q>, input_scale: f64, input_zero: i64,
    output_scale: f64, output_zero: i64, minimum: i64, maximum: i64, #[comptime] relu: bool,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let value = affine_value(input[address(input, p)], input_scale, input_zero);
        let mut q = affine_integer(value, output_scale, output_zero, minimum, maximum);
        if comptime!(relu) { if q < output_zero { q = output_zero; } }
        output[address(output, p)] = Q::cast_from(q);
    }
}

#[ruda(launch)]
fn prelu<Q: Numeric + RudaElement>(
    input: &Tensor<Q>, weight: &Tensor<Q>, output: &mut Tensor<Q>,
    input_scale: f32, weight_scale: f32, output_scale: f32,
    input_zero: i64, weight_zero: i64, output_zero: i64, quantized_one: i64, minimum: i64, maximum: i64,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let x = i64::cast_from(input[address(input, p)]);
        let w = i32::cast_from(weight[address(weight, p)]);
        let mut positive = input_zero; let mut negative = input_zero;
        if x > input_zero { positive = x; } else { negative = x; }
        let positive_difference = u32::cast_from(i32::cast_from(positive)) - u32::cast_from(i32::cast_from(input_zero));
        let negative_difference = u32::cast_from(i32::cast_from(negative)) - u32::cast_from(i32::cast_from(input_zero));
        let one_difference = u32::cast_from(i32::cast_from(quantized_one)) - u32::cast_from(i32::cast_from(weight_zero));
        let weight_difference = u32::cast_from(w) - u32::cast_from(i32::cast_from(weight_zero));
        let sum = i32::cast_from(positive_difference * one_difference + negative_difference * weight_difference);
        let multiplier = input_scale * weight_scale * (1.0f32 / output_scale);
        let scaled = f32::cast_from(f64::cast_from(sum) * f64::cast_from(multiplier));
        let q = i64::cast_from(f64::cast_from(scaled.round()) + f64::cast_from(output_zero));
        output[address(output, p)] = Q::cast_from(clamp_integer(q, minimum, maximum));
    }
}

#[ruda]
fn row_offset<Q: Numeric>(tensor: &Tensor<Q>, row: usize, #[comptime] axis: usize) -> usize {
    let mut remaining = row; let mut base = 0usize; let mut dim = tensor.rank();
    while dim > 0 {
        dim -= 1;
        if dim != axis {
            base += remaining % tensor.shape(dim) * tensor.stride(dim);
            remaining /= tensor.shape(dim);
        }
    }
    base
}

#[ruda(launch)]
fn softmax<Q: Numeric + RudaElement>(
    input: &Tensor<Q>, output: &mut Tensor<Q>, input_scale: f64, input_zero: i64,
    output_scale: f64, output_zero: i64, minimum: i64, maximum: i64, #[comptime] axis: usize,
) {
    let row = RUDA_POS_X as usize * 4usize + UNIT_POS as usize / 32usize;
    let lane = UNIT_POS as usize % 32usize;
    let width = input.shape(axis);
    if row < input.len() / width {
        let ib = row_offset(input, row, axis); let ob = row_offset(output, row, axis);
        let mut largest = f32::cast_from(f32::NEG_INFINITY); let mut nan_count = 0.0f32; let mut i = lane;
        while i < width {
            let value = affine_value(input[ib + i * input.stride(axis)], input_scale, input_zero);
            if value != value { nan_count = 1.0f32; }
            if value > largest { largest = value; }
            i += 32usize;
        }
        largest = plane_max(largest);
        nan_count = plane_sum(nan_count);
        if nan_count > 0.0f32 { largest = f32::NAN; }
        let mut total = 0.0f32; i = lane;
        while i < width {
            let value = affine_value(input[ib + i * input.stride(axis)], input_scale, input_zero);
            total += (value - largest).exp(); i += 32usize;
        }
        total = plane_sum(total); i = lane;
        while i < width {
            let value = affine_value(input[ib + i * input.stride(axis)], input_scale, input_zero);
            let probability = (value - largest).exp() / total;
            let q = affine_integer(probability, output_scale, output_zero, minimum, maximum);
            output[ob + i * output.stride(axis)] = Q::cast_from(q);
            i += 32usize;
        }
    }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64], values: &[f64]) {
    assert!((556..=558).contains(&op));
    assert_eq!(xs.len(), if op == 557 { 3 } else { 2 });
    assert_eq!(ints.len(), if op == 557 { 4 } else { 3 });
    assert_eq!(values.len(), if op == 557 { 3 } else { 2 });
    let output = xs.last().unwrap();
    for x in xs { assert_eq!(x.dtype, output.dtype); assert_eq!(x.shape, output.shape); }
    let (minimum, maximum): (i64, i64) = match output.dtype {
        5 => (i32::MIN as i64, i32::MAX as i64), 7 => (-128, 127), 8 => (0, 255),
        _ => panic!("quantized operator storage must be int8, uint8 or int32"),
    };
    assert!((minimum..=maximum).contains(&ints[0]));
    assert!((minimum..=maximum).contains(&ints[1]));
    let mut axis = 0usize;
    if op == 556 { assert!((0..=1).contains(&ints[2])); }
    else if op == 557 {
        assert!((minimum..=maximum).contains(&ints[2]));
        assert!((minimum..=maximum).contains(&ints[3]));
    }
    else {
        axis = usize::try_from(ints[2]).expect("negative softmax axis");
        assert!(axis < output.shape.len());
    }
    if output.len == 0 { return; }
    let c = client();
    let block = RudaDim::new_1d(if op == 558 { 128 } else { 256 });
    let blocks = if op == 558 { (output.len / output.shape[axis]).div_ceil(4) } else { output.len.div_ceil(256) };
    let grid = RudaCount::Static(u32::try_from(blocks).expect("quantized operator launch overflow"), 1, 1);
    macro_rules! run { ($q:ty) => { unsafe {
        match op {
            556 => concatenate_segment::launch::<$q,CudaRuntime>(&c,grid,block,xs[0].arg(),output.arg(),
                values[0],ints[0],values[1],ints[1],minimum,maximum,ints[2]!=0),
            557 => prelu::launch::<$q,CudaRuntime>(&c,grid,block,xs[0].arg(),xs[1].arg(),output.arg(),
                values[0] as f32,values[1] as f32,values[2] as f32,ints[0],ints[1],ints[2],ints[3],minimum,maximum),
            558 => softmax::launch::<$q,CudaRuntime>(&c,grid,block,xs[0].arg(),output.arg(),
                values[0],ints[0],values[1],ints[1],minimum,maximum,axis),
            _ => unreachable!(),
        };
    } }; }
    match output.dtype { 5=>run!(i32),7=>run!(i8),8=>run!(u8),_=>unreachable!() }
}
