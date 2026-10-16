use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn quantized_integer(value: f32, zero: i64, minimum: i64, maximum: i64) -> i64 {
    let mut q = f64::cast_from(value.round()) + f64::cast_from(zero);
    if q < f64::cast_from(minimum) { q = f64::cast_from(minimum); }
    if q > f64::cast_from(maximum) { q = f64::cast_from(maximum); }
    i64::cast_from(q)
}

#[ruda(launch)]
fn batch_parameters(
    weight: &Tensor<f32>, bias: &Tensor<f32>, mean: &Tensor<f32>, variance: &Tensor<f32>,
    alpha: &mut Tensor<f32>, beta: &mut Tensor<f32>, epsilon: f64, input_scale: f64, output_scale: f64,
) {
    let channel = ABSOLUTE_POS as usize;
    if channel < alpha.len() {
        let w = weight[address(weight, channel)]; let b = bias[address(bias, channel)];
        let m = mean[address(mean, channel)]; let v = variance[address(variance, channel)];
        let inverse = f32::cast_from(1.0f64 / f64::cast_from((v + f32::cast_from(epsilon)).sqrt()));
        alpha[address(alpha, channel)] = f32::cast_from(f64::cast_from(inverse * w) * (input_scale / output_scale));
        beta[address(beta, channel)] = f32::cast_from(f64::cast_from(b - m * inverse * w) / output_scale);
    }
}

#[ruda(launch)]
fn batch_apply<Q: Numeric + RudaElement>(
    input: &Tensor<Q>, alpha: &Tensor<f32>, beta: &Tensor<f32>, output: &mut Tensor<Q>,
    input_zero: i64, output_zero: i64, minimum: i64, maximum: i64,
    #[comptime] channels: usize, #[comptime] spatial: usize, #[comptime] relu: bool,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let channel = p / spatial % channels;
        let difference = i64::cast_from(input[address(input, p)]) - input_zero;
        let value = alpha[address(alpha, channel)] * f32::cast_from(difference) + beta[address(beta, channel)];
        let mut q = quantized_integer(value, output_zero, minimum, maximum);
        if comptime!(relu) { if q < output_zero { q = output_zero; } }
        output[address(output, p)] = Q::cast_from(q);
    }
}

#[ruda(launch)]
fn normalize<Q: Numeric + RudaElement>(
    input: &Tensor<Q>, weight: &Tensor<f32>, bias: &Tensor<f32>, output: &mut Tensor<Q>,
    epsilon: f64, input_scale: f32, output_scale: f32, input_zero: i64, output_zero: i64,
    minimum: i64, maximum: i64, #[comptime] rows: usize, #[comptime] width: usize,
    #[comptime] channels: usize, #[comptime] spatial: usize, #[comptime] per_channel: bool,
    #[comptime] channels_last: bool, #[comptime] has_weight: bool, #[comptime] has_bias: bool,
    #[comptime] float_squares: bool,
) {
    let row = RUDA_POS_X as usize * 4usize + UNIT_POS as usize / 32usize;
    let lane = UNIT_POS as usize % 32usize;
    if row < rows {
        let mut sum = 0i64; let mut squares = 0i64; let mut squares_f = 0.0f32;
        let mut i = lane;
        while i < width {
            let value = i64::cast_from(input[address(input, row * width + i)]);
            sum += value;
            if comptime!(float_squares) {
                let real = f32::cast_from(value); squares_f += real * real;
            } else { squares += value * value; }
            i += 32usize;
        }
        let raw_mean = f32::cast_from(plane_sum(sum)) / f32::cast_from(width);
        let mut square_mean = 0.0f32;
        if comptime!(float_squares) { square_mean = plane_sum(squares_f) / f32::cast_from(width); }
        else { square_mean = f32::cast_from(plane_sum(squares)) / f32::cast_from(width); }
        let mean = raw_mean - f32::cast_from(input_zero);
        let mut variance = square_mean - raw_mean * raw_mean;
        if variance < 0.0f32 { variance = 0.0f32; }
        let denominator = (f64::cast_from(variance * input_scale * input_scale) + epsilon).sqrt();
        let inverse = f32::cast_from(f64::cast_from(input_scale) / denominator);
        let output_inverse = 1.0f32 / output_scale;
        i = lane;
        while i < width {
            let p = row * width + i;
            let mut parameter = i;
            if comptime!(per_channel) { parameter = p / spatial % channels; }
            let mut gamma = 1.0f32; let mut beta = 0.0f32;
            if comptime!(has_weight) { gamma = weight[address(weight, parameter)]; }
            if comptime!(has_bias) { beta = bias[address(bias, parameter)]; }
            let shifted = f32::cast_from(input[address(input, p)]) - f32::cast_from(input_zero);
            let mut value = 0.0f32;
            if comptime!(per_channel) {
                let gamma_p = inverse * gamma;
                if comptime!(channels_last) { value = shifted * gamma_p + (-gamma_p * mean + beta); }
                else { value = (shifted - mean) * gamma_p + beta; }
            } else { value = ((shifted - mean) * inverse) * gamma + beta; }
            output[address(output, p)] = Q::cast_from(quantized_integer(value * output_inverse, output_zero, minimum, maximum));
            i += 32usize;
        }
    }
}

fn grid(n: usize) -> RudaCount {
    RudaCount::Static(u32::try_from(n).expect("quantized normalization launch overflow"), 1, 1)
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64], values: &[f64]) {
    assert!((559..=561).contains(&op));
    let c = client();
    if op == 559 {
        assert_eq!(xs.len(), 6); assert!(ints.is_empty()); assert_eq!(values.len(), 3);
        for x in xs { assert_eq!(x.dtype, 0); assert_eq!(x.len, xs[0].len); }
        if xs[0].len != 0 { unsafe {
            batch_parameters::launch::<CudaRuntime>(&c,grid(xs[0].len.div_ceil(256)),RudaDim::new_1d(256),
                xs[0].arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),xs[4].arg(),xs[5].arg(),
                values[0],values[1],values[2]);
        } }
        return;
    }
    assert_eq!(xs.len(), 4); assert_eq!(xs[0].dtype, xs[3].dtype); assert_eq!(xs[0].shape, xs[3].shape);
    assert_eq!(xs[1].dtype, 0); assert_eq!(xs[2].dtype, 0);
    let (minimum, maximum): (i64, i64) = match xs[0].dtype {
        5 => (i32::MIN as i64, i32::MAX as i64), 7 => (-128, 127), 8 => (0, 255),
        _ => panic!("quantized normalization storage must be int8, uint8 or int32"),
    };
    if op == 560 {
        assert_eq!(ints.len(), 5); assert!(values.is_empty()); assert!((0..=1).contains(&ints[4]));
        assert!((minimum..=maximum).contains(&ints[0]) && (minimum..=maximum).contains(&ints[1]));
        let channels = usize::try_from(ints[2]).expect("negative channels");
        let spatial = usize::try_from(ints[3]).expect("negative spatial size");
        assert_eq!(xs[1].len, channels); assert_eq!(xs[2].len, channels);
        assert!(xs[0].shape.len() >= 2); assert_eq!(xs[0].shape[1], channels);
        assert_eq!(xs[0].shape[2..].iter().product::<usize>(), spatial);
        if xs[0].len == 0 { return; }
        let work = grid(xs[0].len.div_ceil(256)); let block = RudaDim::new_1d(256);
        macro_rules! run { ($q:ty) => { unsafe {
            batch_apply::launch::<$q,CudaRuntime>(&c,work,block,xs[0].arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),
                ints[0],ints[1],minimum,maximum,channels,spatial,ints[4]!=0);
        } }; }
        match xs[0].dtype { 5=>run!(i32),7=>run!(i8),8=>run!(u8),_=>unreachable!() }
    } else {
        assert_eq!(ints.len(), 8); assert_eq!(values.len(), 3);
        assert!((0..=1).contains(&ints[0]) && (0..=1).contains(&ints[7]));
        let rows = usize::try_from(ints[1]).expect("negative normalization row count");
        let width = usize::try_from(ints[2]).expect("negative normalization row width");
        let channels = usize::try_from(ints[3]).expect("negative normalization channel count");
        let spatial = usize::try_from(ints[4]).expect("negative normalization spatial size");
        assert_eq!(rows.checked_mul(width).expect("normalization element overflow"), xs[0].len);
        assert!((minimum..=maximum).contains(&ints[5]) && (minimum..=maximum).contains(&ints[6]));
        let affine = if ints[0] == 0 { width } else { channels };
        assert!(xs[1].len == 0 || xs[1].len == affine); assert!(xs[2].len == 0 || xs[2].len == affine);
        if xs[0].len == 0 { return; }
        if ints[0] == 1 {
            assert!(channels > 0 && spatial > 0);
            assert!(xs[0].shape.len() >= 3 && xs[0].shape[1] == channels);
            assert_eq!(xs[0].shape[2..].iter().product::<usize>(), spatial);
        }
        let work = grid(rows.div_ceil(4)); let block = RudaDim::new_1d(128);
        macro_rules! run { ($q:ty) => { unsafe {
            normalize::launch::<$q,CudaRuntime>(&c,work,block,xs[0].arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),
                values[0],values[1] as f32,values[2] as f32,ints[5],ints[6],minimum,maximum,
                rows,width,channels,spatial,ints[0]!=0,ints[7]!=0,xs[1].len!=0,xs[2].len!=0,xs[0].dtype==5);
        } }; }
        match xs[0].dtype { 5=>run!(i32),7=>run!(i8),8=>run!(u8),_=>unreachable!() }
    }
}
