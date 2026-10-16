// ATen FunctionOfAMatrixUtils accumulation semantics; PyTorch license in bessel.rs.
use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn floating<S: Float + RudaElement, F: Float>(
    input: &Tensor<S>, coefficients: &Tensor<S>, output: &mut Tensor<S>,
    input_stride: usize, coefficient_stride: usize, count: usize,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < output.len() {
        let source = address(input, p); let weights = address(coefficients, p); let target = address(output, p);
        let mut value = output[target];
        for j in 0..count {
            let product = S::cast_from(F::cast_from(input[source + j * input_stride]) * F::cast_from(coefficients[weights + j * coefficient_stride]));
            value = S::cast_from(F::cast_from(value) + F::cast_from(product));
        }
        output[target] = value;
    }
}

#[ruda(launch)]
fn integer<S: Numeric + RudaElement>(
    input: &Tensor<S>, coefficients: &Tensor<S>, output: &mut Tensor<S>,
    input_stride: usize, coefficient_stride: usize, count: usize, #[comptime] boolean: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < output.len() {
        let source = address(input, p); let weights = address(coefficients, p); let target = address(output, p);
        let mut value = u64::cast_from(output[target]);
        for j in 0..count {
            let a = u64::cast_from(input[source + j * input_stride]); let b = u64::cast_from(coefficients[weights + j * coefficient_stride]);
            if comptime!(boolean) { value = u64::cast_from(value != 0u64 || (a != 0u64 && b != 0u64)); }
            else { value += a * b; }
        }
        output[target] = S::cast_from(value);
    }
}

#[ruda(launch)]
fn complex<F: Float + RudaElement>(
    real: &Tensor<F>, imag: &Tensor<F>, coefficients: &Tensor<F>, out_real: &mut Tensor<F>, out_imag: &mut Tensor<F>,
    input_stride: usize, coefficient_stride: usize, count: usize,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < out_real.len() {
        let source_r = address(real, p); let source_i = address(imag, p); let weights = address(coefficients, p);
        let target_r = address(out_real, p); let target_i = address(out_imag, p);
        let mut r = out_real[target_r]; let mut i = out_imag[target_i];
        for j in 0..count {
            let weight = coefficients[weights + j * coefficient_stride];
            r += real[source_r + j * input_stride] * weight; i += imag[source_i + j * input_stride] * weight;
        }
        out_real[target_r] = r; out_imag[target_i] = i;
    }
}

fn summation_span(value: &View, stride: usize, count: usize) {
    scientific::indexable(value);
    if value.len == 0 || count == 0 { return; }
    let span = value.shape.iter().zip(&value.strides).try_fold(1usize, |n, (&size, &step)|
        (size - 1).checked_mul(step).and_then(|part| n.checked_add(part))).expect("linear-combination strided span overflow");
    let span = (count - 1).checked_mul(stride).and_then(|extra| span.checked_add(extra)).expect("linear-combination reduction span overflow");
    assert!(span <= u32::MAX as usize, "linear-combination reduction exceeds U32 indexing");
    let bytes = span.checked_mul(element_bytes(value.dtype)).expect("linear-combination byte span overflow");
    let available = value.handle.size().checked_sub(value.handle.offset_start.unwrap_or(0))
        .and_then(|n| n.checked_sub(value.handle.offset_end.unwrap_or(0))).expect("linear-combination allocation offsets");
    assert!(bytes as u64 <= available, "linear-combination reduction exceeds backing allocation");
}

pub(super) fn dispatch(xs: &[View], ints: &[i64]) {
    assert_eq!(xs.len(), 3); assert_eq!(ints.len(), 3);
    let input_stride = usize::try_from(ints[0]).expect("nonnegative input reduction stride");
    let coefficient_stride = usize::try_from(ints[1]).expect("nonnegative coefficient reduction stride");
    let count = usize::try_from(ints[2]).expect("nonnegative linear-combination reduction length");
    assert!(count <= u32::MAX as usize, "linear-combination reduction length exceeds U32 indexing");
    let code = xs[0].dtype; assert!(code <= 11); assert_eq!(xs[2].dtype, code);
    let coefficient_code = match code { 10 => 0, 11 => 9, _ => code };
    assert_eq!(xs[1].dtype, coefficient_code, "linear-combination coefficients must use the input's real dtype");
    assert_eq!(xs[0].shape, xs[2].shape); assert_eq!(xs[1].shape, xs[2].shape);
    summation_span(&xs[0], input_stride, count); summation_span(&xs[1], coefficient_stride, count);
    if xs[2].len == 0 || count == 0 { return; }
    let c = client(); let blocks = xs[2].len.div_ceil(256); let x = blocks.min(131072);
    let grid = RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1); let dim = RudaDim::new_1d(256);
    if matches!(code, 10 | 11) {
        let ar = scientific::component(&xs[0], false); let ai = scientific::component(&xs[0], true);
        let yr = scientific::component(&xs[2], false); let yi = scientific::component(&xs[2], true);
        let stride = input_stride.checked_mul(2).expect("complex reduction stride overflow");
        summation_span(&ar, stride, count); summation_span(&ai, stride, count);
        scientific::indexable(&yr); scientific::indexable(&yi);
        macro_rules! run { ($f:ty) => { unsafe {
            complex::launch::<$f, CudaRuntime>(&c, grid, dim, ar.arg(), ai.arg(), xs[1].arg(), yr.arg(), yi.arg(), stride, coefficient_stride, count);
        }}; }
        match code { 10 => run!(f32), 11 => run!(f64), _ => unreachable!() }
    } else {
        macro_rules! floating { ($s:ty, $f:ty) => { unsafe {
            floating::launch::<$s, $f, CudaRuntime>(&c, grid, dim, xs[0].arg(), xs[1].arg(), xs[2].arg(), input_stride, coefficient_stride, count);
        }}; }
        macro_rules! integer { ($s:ty) => { unsafe {
            integer::launch::<$s, CudaRuntime>(&c, grid, dim, xs[0].arg(), xs[1].arg(), xs[2].arg(), input_stride, coefficient_stride, count, code == 3);
        }}; }
        match code {
            0 => floating!(f32, f32), 1 => floating!(f16, f32), 2 => floating!(bf16, f32), 9 => floating!(f64, f64),
            3 | 8 => integer!(u8), 4 => integer!(i64), 5 => integer!(i32), 6 => integer!(i16), 7 => integer!(i8),
            _ => unreachable!(),
        }
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed); finish_dispatch(&c);
}
