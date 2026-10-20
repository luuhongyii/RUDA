use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn cast_input<S: Float, O: Float, F: Float>(value: S, #[comptime] output_code: u32) -> F {
    let mut result = F::new(0.0);
    if comptime!(output_code == 1 || output_code == 2) {
        result = F::cast_from(O::cast_from(f32::cast_from(value)));
    } else { result = F::cast_from(O::cast_from(value)); }
    result
}

#[ruda]
fn identity<F: Float>(#[comptime] kind: u32) -> F {
    let mut value = F::new(0.0);
    if comptime!(kind == 4) { value = F::new(f32::INFINITY); }
    value
}

#[ruda]
fn combine<F: Float>(a: F, b: F, #[comptime] kind: u32) -> F {
    let mut value = a;
    if comptime!(kind == 3) { if b != b || a < b { value = b; } }
    else if comptime!(kind == 4) { if b != b || b < a { value = b; } }
    else { value = a + b; }
    value
}

#[ruda]
fn block_reduce<F: Float>(value: F, #[comptime] kind: u32) -> F {
    let tid = UNIT_POS as usize;
    let mut result = value;
    let mut offset = 16u32;
    while offset > 0u32 {
        result = combine::<F>(result, plane_shuffle_down(result, offset), kind);
        offset /= 2u32;
    }
    let mut shared = SharedMemory::<F>::new(8usize);
    sync_ruda();
    if tid % 32usize == 0 { shared[tid / 32usize] = result; }
    sync_ruda();
    result = identity::<F>(kind);
    if tid < 8usize { result = shared[tid]; }
    if tid < 32usize {
        offset = 16u32;
        while offset > 0u32 {
            result = combine::<F>(result, plane_shuffle_down(result, offset), kind);
            offset /= 2u32;
        }
    }
    result
}

#[ruda]
fn term<F: Float>(real: F, imag: F, order: F, #[comptime] kind: u32, #[comptime] complex: bool) -> F {
    let mut value = real.abs();
    if comptime!(kind == 0) {
        value = F::cast_from(real != F::new(0.0) || imag != F::new(0.0));
    } else {
        if comptime!(complex) { value = real.hypot(imag); }
        if comptime!(kind == 2) {
            if comptime!(!complex) { value = real; }
            value *= value;
        } else if comptime!(kind == 5) { value = value.powf(order); }
    }
    value
}

#[ruda]
fn project<F: Float>(value: F, order: F, #[comptime] kind: u32, #[comptime] power_sum: bool) -> F {
    let mut result = value;
    if comptime!(!power_sum) {
        if comptime!(kind == 2) { result = value.sqrt(); }
        else if comptime!(kind == 5) { result = value.powf(F::new(1.0) / order); }
    }
    result
}

#[ruda(launch)]
fn empty<O: Float + RudaElement>(out: &mut Tensor<O>) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < out.len() { out[p] = O::new(0.0); }
}

#[ruda(launch)]
fn singleton<S: Float + RudaElement, O: Float + RudaElement, F: Float>(
    real: &Tensor<S>, imag: &Tensor<S>, out: &mut Tensor<O>,
    #[comptime] zero_order: bool, #[comptime] complex: bool, #[comptime] output_code: u32,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < out.len() {
        let x = cast_input::<S, O, F>(real[address(real, p)], output_code);
        let mut y = F::new(0.0);
        if comptime!(complex) { y = cast_input::<S, O, F>(imag[address(imag, p)], output_code); }
        let mut result = x.abs();
        if comptime!(zero_order) { result = F::cast_from(x != F::new(0.0) || y != F::new(0.0)); }
        else if comptime!(complex) { result = x.hypot(y); }
        out[p] = O::cast_from(result);
    }
}

#[ruda(launch)]
fn chunks<S: Float + RudaElement, O: Float + RudaElement, F: Float + RudaElement>(
    real: &Tensor<S>, imag: &Tensor<S>, partials: &mut Tensor<F>, out: &mut Tensor<O>,
    width: usize, count: usize, order: F,
    #[comptime] kind: u32, #[comptime] power_sum: bool, #[comptime] complex: bool,
    #[comptime] output_code: u32, #[comptime] last: bool,
) {
    let task = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    let tid = UNIT_POS as usize;
    if task < out.len() * count {
        let row = task / count;
        let chunk = task % count;
        let start = chunk * 65536usize;
        let remaining = width - start;
        let mut value = identity::<F>(kind);
        let mut offset = tid;
        while offset < remaining && offset < 65536usize {
            let p = row * width + start + offset;
            let x = cast_input::<S, O, F>(real[address(real, p)], output_code);
            let mut y = F::new(0.0);
            if comptime!(complex) { y = cast_input::<S, O, F>(imag[address(imag, p)], output_code); }
            value = combine::<F>(value, term::<F>(x, y, order, kind, complex), kind);
            offset += 256usize;
        }
        value = block_reduce::<F>(value, kind);
        if tid == 0 {
            if comptime!(last) { out[row] = O::cast_from(project::<F>(value, order, kind, power_sum)); }
            else { partials[task] = value; }
        }
    }
}

#[ruda(launch)]
fn finish<O: Float + RudaElement, F: Float + RudaElement>(
    partials: &Tensor<F>, out: &mut Tensor<O>, count: usize, order: F,
    #[comptime] kind: u32, #[comptime] power_sum: bool,
) {
    let row = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    let tid = UNIT_POS as usize;
    if row < out.len() {
        let mut value = identity::<F>(kind);
        let mut chunk = tid;
        while chunk < count {
            value = combine::<F>(value, partials[row * count + chunk], kind);
            chunk += 256usize;
        }
        value = block_reduce::<F>(value, kind);
        if tid == 0 { out[row] = O::cast_from(project::<F>(value, order, kind, power_sum)); }
    }
}

fn grid(tasks: usize) -> RudaCount {
    assert!(tasks > 0 && tasks <= u32::MAX as usize, "norm launch exceeds U32 indexing");
    let x = tasks.min(131072);
    RudaCount::Static(x as u32, tasks.div_ceil(x) as u32, 1)
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64], values: &[f64]) {
    assert!(matches!(op, 455 | 456));
    assert_eq!(xs.len(), 3);
    assert_eq!(ints.len(), 2);
    assert_eq!(values.len(), 1);
    let power_sum = op == 456;
    let real = &xs[0];
    let imag = &xs[1];
    let out = &xs[2];
    assert!(ints[0] >= 0 && ints[0] as usize <= real.shape.len() && matches!(ints[1], 0 | 1));
    let outer_rank = ints[0] as usize;
    let complex = ints[1] != 0;
    assert!(matches!(real.dtype, 0 | 1 | 2 | 9) && matches!(out.dtype, 0 | 1 | 2 | 9));
    assert_eq!(imag.dtype, real.dtype);
    assert_eq!(imag.shape, real.shape);
    let product = |shape: &[usize]| shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)).expect("norm shape overflow");
    let rows = product(&real.shape[..outer_rank]);
    let width = product(&real.shape[outer_rank..]);
    assert_eq!(out.shape, vec![rows]);
    assert_eq!(out.strides, vec![1usize]);
    assert_eq!(real.len, rows.checked_mul(width).expect("norm element count overflow"));
    let order = values[0];
    if !power_sum && width == 0 { assert!(!(order < 0.0 || order == f64::INFINITY)); }
    if rows == 0 { return; }
    let kind = if power_sum { if order == 2.0 { 2 } else { 5 } }
        else if order == 0.0 { 0 } else if order == 1.0 { 1 } else if order == 2.0 { 2 }
        else if order == f64::INFINITY { 3 } else if order == f64::NEG_INFINITY { 4 } else { 5 };
    let c = client();
    let block = RudaDim::new_1d(256);
    if width == 0 {
        macro_rules! fill { ($o:ty) => { unsafe { empty::launch::<$o, CudaRuntime>(&c, grid(rows.div_ceil(256)), block, out.arg()); }}; }
        match out.dtype { 0 => fill!(f32), 1 => fill!(f16), 2 => fill!(bf16), 9 => fill!(f64), _ => unreachable!() }
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
        finish_dispatch(&c);
        return;
    }
    let count = width.div_ceil(65536);
    let tasks = rows.checked_mul(count).expect("norm partial count overflow");
    let partial_code = if out.dtype == 9 { 9 } else { 0 };
    let partial_len = if count > 1 { tasks } else { 1 };
    assert!(tasks <= u32::MAX as usize);
    let partials = View::packed(c.empty(partial_len.checked_mul(element_bytes(partial_code)).expect("norm workspace overflow")),
        vec![partial_len], partial_code);
    macro_rules! run { ($s:ty, $o:ty, $f:ty) => { unsafe {
        if width == 1 && !power_sum {
            singleton::launch::<$s, $o, $f, CudaRuntime>(&c, grid(rows.div_ceil(256)), block,
                real.arg(), imag.arg(), out.arg(), kind == 0, complex, out.dtype);
        } else {
            chunks::launch::<$s, $o, $f, CudaRuntime>(&c, grid(tasks), block, real.arg(), imag.arg(), partials.arg(), out.arg(),
                width, count, order as $f, kind, power_sum, complex, out.dtype, count == 1);
            if count > 1 {
                finish::launch::<$o, $f, CudaRuntime>(&c, grid(rows), block, partials.arg(), out.arg(), count, order as $f, kind, power_sum);
                LAUNCHES.fetch_add(1, Ordering::Relaxed);
            }
        }
    }}; }
    macro_rules! target { ($s:ty) => { match out.dtype {
        0 => run!($s, f32, f32), 1 => run!($s, f16, f32), 2 => run!($s, bf16, f32),
        9 => run!($s, f64, f64), _ => unreachable!(),
    }}; }
    match real.dtype { 0 => target!(f32), 1 => target!(f16), 2 => target!(bf16), 9 => target!(f64), _ => unreachable!() }
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    finish_dispatch(&c);
}
