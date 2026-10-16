use super::*;
use super::blas_accumulate::block_sum;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn chunks<S: Float + RudaElement, F: Float + RudaElement>(
    left: &Tensor<S>, right: &Tensor<S>, partials: &mut Tensor<F>, out: &mut Tensor<S>,
    count: usize, #[comptime] last: bool,
) {
    let chunk = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    if chunk >= count { return; }
    let tid = UNIT_POS as usize; let start = chunk * 4096usize;
    let remaining = left.len() - start;
    let mut sum = F::new(0.0);
    for step in 0..16usize {
        let offset = step * 256usize + tid;
        if offset < remaining {
            let p = start + offset;
            sum = fma(F::cast_from(left[p * left.stride(0)]), F::cast_from(right[p * right.stride(0)]), sum);
        }
    }
    let sum = block_sum(sum);
    if tid == 0usize {
        if comptime!(last) { out[0usize] = S::cast_from(sum); }
        else { partials[chunk] = sum; }
    }
}

#[ruda(launch)]
fn finish<S: Float + RudaElement, F: Float + RudaElement>(partials: &Tensor<F>, out: &mut Tensor<S>) {
    let tid = UNIT_POS as usize;
    let mut sum = F::new(0.0); let mut i = tid;
    while i < partials.len() { sum += partials[i]; i += 256usize; }
    let sum = block_sum(sum);
    if tid == 0usize { out[0usize] = S::cast_from(sum); }
}

#[ruda(launch)]
fn complex_chunks<F: Float + RudaElement>(
    ar: &Tensor<F>, ai: &Tensor<F>, br: &Tensor<F>, bi: &Tensor<F>, partials: &mut Tensor<F>,
    yr: &mut Tensor<F>, yi: &mut Tensor<F>, count: usize,
    #[comptime] conjugate: bool, #[comptime] last: bool,
) {
    let chunk = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    if chunk >= count { return; }
    let tid = UNIT_POS as usize; let start = chunk * 4096usize;
    let remaining = ar.len() - start;
    let mut real = F::new(0.0); let mut imag = F::new(0.0);
    for step in 0..16usize {
        let offset = step * 256usize + tid;
        if offset < remaining {
            let p = start + offset;
            let x = ar[p * ar.stride(0)]; let mut u = ai[p * ai.stride(0)];
            let y = br[p * br.stride(0)]; let v = bi[p * bi.stride(0)];
            if comptime!(conjugate) { u = -u; }
            real += x * y - u * v; imag += x * v + u * y;
        }
    }
    let real = block_sum(real); let imag = block_sum(imag);
    if tid == 0usize {
        if comptime!(last) { yr[0usize] = real; yi[0usize] = imag; }
        else { partials[2usize * chunk] = real; partials[2usize * chunk + 1usize] = imag; }
    }
}

#[ruda(launch)]
fn complex_finish<F: Float + RudaElement>(partials: &Tensor<F>, yr: &mut Tensor<F>, yi: &mut Tensor<F>) {
    let tid = UNIT_POS as usize;
    let mut real = F::new(0.0); let mut imag = F::new(0.0); let mut i = tid;
    while i < partials.len() / 2usize {
        real += partials[2usize * i]; imag += partials[2usize * i + 1usize]; i += 256usize;
    }
    let real = block_sum(real); let imag = block_sum(imag);
    if tid == 0usize { yr[0usize] = real; yi[0usize] = imag; }
}

pub(super) fn dispatch(xs: &[View], ints: &[i64]) {
    assert_eq!(xs.len(), 3); assert_eq!(ints.len(), 1); assert!(ints[0] == 0 || ints[0] == 1);
    let (a, b, out) = (&xs[0], &xs[1], &xs[2]);
    assert_eq!(a.shape.len(), 1); assert_eq!(a.shape, b.shape); assert!(out.shape.is_empty());
    assert_eq!(a.dtype, b.dtype); assert_eq!(out.dtype, a.dtype);
    assert!(matches!(a.dtype, 0 | 1 | 2 | 9 | 10 | 11));
    assert!(a.len <= i32::MAX as usize && a.strides[0] <= i32::MAX as usize && b.strides[0] <= i32::MAX as usize,
        "dot requires length and input strides within signed int32");
    let count = a.len.div_ceil(4096).max(1); let complex = matches!(a.dtype, 10 | 11);
    let partial_code = if matches!(a.dtype, 9 | 11) { 9 } else { 0 };
    let width = if complex { 2 } else { 1 };
    let c = client();
    let partials = View::packed(c.empty(count * width * element_bytes(partial_code)), vec![count * width], partial_code);
    let x = count.min(131072); let grid = RudaCount::Static(x as u32, count.div_ceil(x) as u32, 1);
    let dim = RudaDim::new_1d(256);
    if complex {
        let parts: Vec<View> = xs.iter().flat_map(|v| [scientific::component(v, false), scientific::component(v, true)]).collect();
        for part in &parts { scientific::indexable(part); }
        macro_rules! run { ($f:ty) => { unsafe {
            complex_chunks::launch::<$f, CudaRuntime>(&c, grid, dim,
                parts[0].arg(), parts[1].arg(), parts[2].arg(), parts[3].arg(), partials.arg(), parts[4].arg(), parts[5].arg(),
                count, ints[0] != 0, count == 1);
            if count > 1 {
                complex_finish::launch::<$f, CudaRuntime>(&c, RudaCount::Static(1, 1, 1), dim,
                    partials.arg(), parts[4].arg(), parts[5].arg());
            }
        }}; }
        match a.dtype { 10 => run!(f32), 11 => run!(f64), _ => unreachable!() }
    } else {
        macro_rules! run { ($s:ty, $f:ty) => { unsafe {
            chunks::launch::<$s, $f, CudaRuntime>(&c, grid, dim, a.arg(), b.arg(), partials.arg(), out.arg(), count, count == 1);
            if count > 1 {
                finish::launch::<$s, $f, CudaRuntime>(&c, RudaCount::Static(1, 1, 1), dim, partials.arg(), out.arg());
            }
        }}; }
        match a.dtype { 0 => run!(f32, f32), 1 => run!(f16, f32), 2 => run!(bf16, f32), 9 => run!(f64, f64), _ => unreachable!() }
    }
    LAUNCHES.fetch_add(if count > 1 { 2 } else { 1 }, Ordering::Relaxed); finish_dispatch(&c);
}
