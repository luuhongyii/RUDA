// ATen GLU formulas; retained PyTorch license: bessel.rs.
use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn gated<S: Float + RudaElement, F: Float + RudaElement>(
    a: &Tensor<S>, b: &Tensor<S>, grad: &Tensor<S>,
    out_a: &mut Tensor<S>, out_b: &mut Tensor<S>, #[comptime] backward: bool,
) {
    let p = ABSOLUTE_POS as usize;
    if p < out_a.len() {
        let x = F::cast_from(a[address(a, p)]);
        let gate = F::cast_from(b[address(b, p)]);
        let one = F::new(1.0);
        let sigmoid = one / (one + (-gate).exp());
        if comptime!(backward) {
            let g = F::cast_from(grad[address(grad, p)]);
            out_a[address(out_a, p)] = S::cast_from(sigmoid * g);
            out_b[address(out_b, p)] = S::cast_from((one - sigmoid) * sigmoid * g * x);
        } else {
            out_a[address(out_a, p)] = S::cast_from(x * sigmoid);
        }
    }
}

#[ruda(launch)]
fn gated_jvp<S: Float + RudaElement, F: Float + RudaElement>(
    result: &Tensor<S>, b: &Tensor<S>, da: &Tensor<S>, db: &Tensor<S>, out: &mut Tensor<S>,
) {
    let p = ABSOLUTE_POS as usize;
    if p < out.len() {
        let r = F::cast_from(result[address(result, p)]);
        let gate = F::cast_from(b[address(b, p)]);
        let dx = F::cast_from(da[address(da, p)]);
        let dg = F::cast_from(db[address(db, p)]);
        let one = F::new(1.0);
        let sigmoid = one / (one + (-gate).exp());
        out[address(out, p)] = S::cast_from(dx * sigmoid + r * (dg - sigmoid * dg));
    }
}

pub(super) fn dispatch(op: u32, xs: &[View]) {
    assert!(matches!(op, 407..=409));
    assert_eq!(xs.len(), if op == 407 { 3 } else { 5 });
    let input = &xs[0];
    for value in &xs[1..] {
        assert_eq!(value.shape, input.shape);
        assert_eq!(value.dtype, input.dtype);
    }
    assert!(matches!(input.dtype, 0 | 1 | 2 | 9));
    if input.len == 0 { return; }
    let c = client();
    let grid = RudaCount::Static(u32::try_from(input.len.div_ceil(256)).expect("activation launch overflow"), 1, 1);
    let block = RudaDim::new_1d(256);
    macro_rules! run { ($s:ty, $f:ty) => { unsafe {
        match op {
            407 => gated::launch::<$s, $f, CudaRuntime>(&c, grid, block,
                xs[0].arg(), xs[1].arg(), xs[0].arg(), xs[2].arg(), xs[2].arg(), false),
            408 => gated::launch::<$s, $f, CudaRuntime>(&c, grid, block,
                xs[0].arg(), xs[1].arg(), xs[2].arg(), xs[3].arg(), xs[4].arg(), true),
            409 => gated_jvp::launch::<$s, $f, CudaRuntime>(&c, grid, block,
                xs[0].arg(), xs[1].arg(), xs[2].arg(), xs[3].arg(), xs[4].arg()),
            _ => unreachable!(),
        }
    }}; }
    match input.dtype {
        0 => run!(f32, f32), 1 => run!(f16, f32), 2 => run!(bf16, f32),
        9 => run!(f64, f64), _ => unreachable!(),
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    finish_dispatch(&c);
}
