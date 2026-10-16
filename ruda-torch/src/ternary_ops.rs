use super::*;
use super::scientific::address;
use super::math_extra::{complex_divide, complex_multiply};
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn real<S: Float + RudaElement, F: Float + RudaElement>(
    a: &Tensor<S>, b: &Tensor<S>, c: &Tensor<S>, alpha: &Tensor<F>, out: &mut Tensor<S>,
    #[comptime] op: u32, #[comptime] scalar_weight: bool,
) {
    let p = ABSOLUTE_POS as usize;
    if p < out.len() {
        let x = F::cast_from(a[address(a, p)]);
        let y = F::cast_from(b[address(b, p)]);
        let z = F::cast_from(c[address(c, p)]);
        let coefficient = alpha[0];
        let one = F::new(1.0);
        let mut result = F::new(0.0);
        if comptime!(op == 401) {
            let mut weight = z;
            if comptime!(scalar_weight) { weight = coefficient; }
            if weight.abs() < F::new(0.5) { result = x + weight * (y - x); }
            else { result = y - (y - x) * (one - weight); }
        } else if comptime!(op == 399) {
            if coefficient == one { result = fma(y, z, x); }
            else { result = fma(coefficient, y * z, x); }
        } else {
            if coefficient == one { result = x + y / z; }
            else { result = fma(coefficient, y / z, x); }
        }
        out[address(out, p)] = S::cast_from(result);
    }
}

#[ruda(launch)]
fn integer<I: Numeric + RudaElement>(
    a: &Tensor<I>, b: &Tensor<I>, c: &Tensor<I>, alpha: &Tensor<i64>, out: &mut Tensor<I>,
) {
    let p = ABSOLUTE_POS as usize;
    if p < out.len() {
        let x = u64::cast_from(i64::cast_from(a[address(a, p)]));
        let y = u64::cast_from(i64::cast_from(b[address(b, p)]));
        let z = u64::cast_from(i64::cast_from(c[address(c, p)]));
        out[address(out, p)] = I::cast_from(x + u64::cast_from(alpha[0]) * (y * z));
    }
}

#[ruda(launch)]
fn complex<S: Float + RudaElement, F: Float + RudaElement>(
    ar: &Tensor<S>, ai: &Tensor<S>, br: &Tensor<S>, bi: &Tensor<S>,
    cr: &Tensor<S>, ci: &Tensor<S>, alpha_r: &Tensor<F>, alpha_i: &Tensor<F>,
    out_r: &mut Tensor<S>, out_i: &mut Tensor<S>,
    #[comptime] op: u32, #[comptime] scalar_weight: bool,
    #[comptime] code: u32, #[comptime] fraction: u32, #[comptime] exponent_bits: u32,
) {
    let p = ABSOLUTE_POS as usize;
    if p < out_r.len() {
        let a = F::cast_from(ar[address(ar, p)]);
        let b = F::cast_from(ai[address(ai, p)]);
        let c = F::cast_from(br[address(br, p)]);
        let d = F::cast_from(bi[address(bi, p)]);
        let mut e = F::cast_from(cr[address(cr, p)]);
        let mut f = F::cast_from(ci[address(ci, p)]);
        let alpha = alpha_r[0];
        let beta = alpha_i[0];
        let mut result_r = F::new(0.0);
        let mut result_i = F::new(0.0);
        if comptime!(op == 401) {
            if comptime!(scalar_weight) { e = alpha; f = beta; }
            let dr = c - a;
            let di = d - b;
            if e * e + f * f < F::new(0.25) {
                let (r, i) = complex_multiply(e, f, dr, di, code);
                result_r = a + r; result_i = b + i;
            } else {
                let (r, i) = complex_multiply(dr, di, F::new(1.0) - e, F::new(0.0) - f, code);
                result_r = c - r; result_i = d - i;
            }
        } else {
            let mut product_r = F::new(0.0);
            let mut product_i = F::new(0.0);
            if comptime!(op == 399) {
                let (r, i) = complex_multiply(c, d, e, f, code);
                product_r = r; product_i = i;
            } else {
                let (r, i) = complex_divide(c, d, e, f, code, fraction, exponent_bits);
                product_r = r; product_i = i;
            }
            let (r, i) = complex_multiply(alpha, beta, product_r, product_i, code);
            result_r = a + r; result_i = b + i;
        }
        out_r[address(out_r, p)] = S::cast_from(result_r);
        out_i[address(out_i, p)] = S::cast_from(result_i);
    }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64]) {
    assert!((399..=401).contains(&op));
    assert_eq!(ints.len(), 2);
    assert!(matches!(ints[0], 0 | 1) && matches!(ints[1], 0 | 1));
    let is_complex = ints[0] != 0;
    let scalar_weight = ints[1] != 0;
    assert!(!scalar_weight || op == 401);
    let components = if is_complex { 2 } else { 1 };
    assert_eq!(xs.len(), 5 * components);
    let code = xs[0].dtype;
    let scalar_code = match code { 1 | 2 => 0, 4..=8 => 4, _ => code };
    for (index, value) in xs.iter().enumerate() {
        if (3 * components..4 * components).contains(&index) {
            assert_eq!(value.len, 1);
            assert_eq!(value.dtype, scalar_code);
        } else {
            assert_eq!(value.dtype, code);
            assert_eq!(value.shape, xs[0].shape);
        }
    }
    if is_complex { assert!(matches!(code, 0 | 9) || (code == 1 && op == 401)); }
    else {
        assert!(matches!(code, 0 | 1 | 2 | 9) || ((4..=8).contains(&code) && op == 399));
    }
    let count = xs[4 * components].len;
    if count == 0 { return; }
    let client = client();
    let grid = RudaCount::Static(u32::try_from(count.div_ceil(256)).expect("ternary launch overflow"), 1, 1);
    let block = RudaDim::new_1d(256);
    macro_rules! floating { ($s:ty, $f:ty, $code:expr, $fraction:expr, $bits:expr) => { unsafe {
        if is_complex {
            complex::launch::<$s, $f, CudaRuntime>(&client, grid, block,
                xs[0].arg(), xs[1].arg(), xs[2].arg(), xs[3].arg(), xs[4].arg(), xs[5].arg(),
                xs[6].arg(), xs[7].arg(), xs[8].arg(), xs[9].arg(), op, scalar_weight, $code, $fraction, $bits);
        } else {
            real::launch::<$s, $f, CudaRuntime>(&client, grid, block,
                xs[0].arg(), xs[1].arg(), xs[2].arg(), xs[3].arg(), xs[4].arg(), op, scalar_weight);
        }
    }}; }
    macro_rules! integral { ($i:ty) => { unsafe {
        integer::launch::<$i, CudaRuntime>(&client, grid, block,
            xs[0].arg(), xs[1].arg(), xs[2].arg(), xs[3].arg(), xs[4].arg());
    }}; }
    match code {
        0 => floating!(f32, f32, 0, 23, 8), 1 => floating!(f16, f32, 0, 23, 8),
        2 => floating!(bf16, f32, 0, 23, 8), 9 => floating!(f64, f64, 9, 52, 11),
        4 => integral!(i64), 5 => integral!(i32), 6 => integral!(i16),
        7 => integral!(i8), 8 => integral!(u8), _ => unreachable!(),
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    finish_dispatch(&client);
}
