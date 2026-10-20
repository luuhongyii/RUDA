use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn real<S: Float + RudaElement, F: Float>(
    input: &Tensor<S>, left: &Tensor<S>, right: &Tensor<S>, output: &mut Tensor<S>,
    alpha_bits: u64, beta_bits: u64, #[comptime] code: u32,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < output.len() {
        let alpha = F::cast_from(crate::floating::from_bits::<S>(alpha_bits, code));
        let beta = F::cast_from(crate::floating::from_bits::<S>(beta_bits, code));
        let x = F::cast_from(left[address(left, p)]);
        let y = F::cast_from(right[address(right, p)]);
        let scaled = F::cast_from(S::cast_from(alpha * x));
        let mut result = F::cast_from(S::cast_from(scaled * y));
        if beta != F::new(0.0) {
            let bias = F::cast_from(S::cast_from(beta * F::cast_from(input[address(input, p)])));
            result = bias + result;
        }
        output[address(output, p)] = S::cast_from(result);
    }
}

#[ruda(launch)]
fn integer<S: Numeric + RudaElement>(
    input: &Tensor<S>, left: &Tensor<S>, right: &Tensor<S>, output: &mut Tensor<S>,
    alpha: u64, beta: u64, #[comptime] boolean: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < output.len() {
        let x = u64::cast_from(left[address(left, p)]);
        let y = u64::cast_from(right[address(right, p)]);
        let mut result = 0u64;
        if comptime!(boolean) {
            let mut value = alpha != 0u64 && x != 0u64 && y != 0u64;
            if beta != 0u64 { value = value || input[address(input, p)] != S::cast_from(0); }
            result = u64::cast_from(value);
        } else {
            result = alpha * x * y;
            if beta != 0u64 { result += beta * u64::cast_from(input[address(input, p)]); }
        }
        output[address(output, p)] = S::cast_from(result);
    }
}

#[ruda(launch)]
fn complex<F: Float + RudaElement>(
    input_real: &Tensor<F>, input_imag: &Tensor<F>, left_real: &Tensor<F>, left_imag: &Tensor<F>,
    right_real: &Tensor<F>, right_imag: &Tensor<F>, output_real: &mut Tensor<F>, output_imag: &mut Tensor<F>,
    alpha_real_bits: u64, alpha_imag_bits: u64, beta_real_bits: u64, beta_imag_bits: u64,
    #[comptime] code: u32,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < output_real.len() {
        let ar = crate::floating::from_bits::<F>(alpha_real_bits, code);
        let ai = crate::floating::from_bits::<F>(alpha_imag_bits, code);
        let br = crate::floating::from_bits::<F>(beta_real_bits, code);
        let bi = crate::floating::from_bits::<F>(beta_imag_bits, code);
        let x = left_real[address(left_real, p)]; let u = left_imag[address(left_imag, p)];
        let y = right_real[address(right_real, p)]; let v = right_imag[address(right_imag, p)];
        let scaled_real = ar * x - ai * u; let scaled_imag = ar * u + ai * x;
        let mut real = scaled_real * y - scaled_imag * v;
        let mut imag = scaled_real * v + scaled_imag * y;
        if br != F::new(0.0) || bi != F::new(0.0) {
            let a = input_real[address(input_real, p)]; let b = input_imag[address(input_imag, p)];
            real = (br * a - bi * b) + real;
            imag = (br * b + bi * a) + imag;
        }
        output_real[address(output_real, p)] = real; output_imag[address(output_imag, p)] = imag;
    }
}

pub(super) fn dispatch(xs: &[View], ints: &[i64]) {
    assert_eq!(xs.len(), 4); assert_eq!(ints.len(), 4);
    let code = xs[3].dtype; assert!(code <= 11);
    for value in &xs[..3] { assert_eq!(value.dtype, code); assert_eq!(value.shape, xs[3].shape); }
    if xs[3].len == 0 { return; }
    let alpha_real = ints[0] as u64; let alpha_imag = ints[1] as u64;
    let beta_real = ints[2] as u64; let beta_imag = ints[3] as u64;
    let c = client(); let blocks = xs[3].len.div_ceil(256); let x = blocks.min(131072);
    let grid = RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1); let block = RudaDim::new_1d(256);
    if matches!(code, 10 | 11) {
        let parts: Vec<View> = xs.iter().flat_map(|value| [scientific::component(value, false), scientific::component(value, true)]).collect();
        for value in &parts { scientific::indexable(value); }
        macro_rules! run { ($f:ty, $code:expr) => { unsafe {
            complex::launch::<$f, CudaRuntime>(&c, grid, block,
                parts[0].arg(), parts[1].arg(), parts[2].arg(), parts[3].arg(),
                parts[4].arg(), parts[5].arg(), parts[6].arg(), parts[7].arg(),
                alpha_real, alpha_imag, beta_real, beta_imag, $code);
        }}; }
        match code { 10 => run!(f32, 0), 11 => run!(f64, 9), _ => unreachable!() }
    } else {
        macro_rules! floating { ($s:ty, $f:ty) => { unsafe {
            real::launch::<$s, $f, CudaRuntime>(&c, grid, block,
                xs[0].arg(), xs[1].arg(), xs[2].arg(), xs[3].arg(), alpha_real, beta_real, code);
        }}; }
        macro_rules! integral { ($s:ty) => { unsafe {
            integer::launch::<$s, CudaRuntime>(&c, grid, block,
                xs[0].arg(), xs[1].arg(), xs[2].arg(), xs[3].arg(), alpha_real, beta_real, code == 3);
        }}; }
        match code {
            0 => floating!(f32, f32), 1 => floating!(f16, f32), 2 => floating!(bf16, f32), 9 => floating!(f64, f64),
            3 | 8 => integral!(u8), 4 => integral!(i64), 5 => integral!(i32), 6 => integral!(i16), 7 => integral!(i8),
            _ => unreachable!(),
        }
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed); finish_dispatch(&c);
}
