use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn clear_status(status: &mut Tensor<u32>) {
    if UNIT_POS == 0u32 { status[0] = 0u32; }
}

#[ruda(launch)]
fn offsets<I: Int + RudaElement>(
    indices: &Tensor<I>, target: &Tensor<u8>, keys: &mut Tensor<i64>, status: &mut Tensor<Atomic<u32>>,
    #[comptime] axis: usize,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < keys.len() {
        let index = i64::cast_from(indices[address(indices, p)]);
        if index < 0i64 || index >= i64::cast_from(target.shape(axis)) {
            status[0].store(1u32);
            keys[p] = 0i64;
        } else {
            let mut remaining = p; let mut stride = 1usize; let mut key = 0usize;
            let mut dim = indices.rank();
            while dim > 0usize {
                dim -= 1usize;
                let mut coordinate = remaining % indices.shape(dim);
                remaining /= indices.shape(dim);
                if dim == axis { coordinate = usize::cast_from(index); }
                key += coordinate * stride;
                stride *= target.shape(dim);
            }
            keys[p] = i64::cast_from(key);
        }
    }
}

#[ruda]
fn identity<S: Numeric>(#[comptime] op: u32, #[comptime] code: u32) -> S {
    let mut value = S::cast_from(0u32);
    if comptime!(op == 1) { value = S::cast_from(1u32); }
    else if comptime!(op == 3 || op == 4) {
        if comptime!(code == 3) { value = S::cast_from(op == 3); }
        else if comptime!(code == 4) {
            if comptime!(op == 3) { value = S::cast_from(i64::MAX); } else { value = S::cast_from(i64::MIN); }
        } else if comptime!(code == 5) {
            if comptime!(op == 3) { value = S::cast_from(i32::MAX); } else { value = S::cast_from(i32::MIN); }
        } else if comptime!(code == 6) {
            if comptime!(op == 3) { value = S::cast_from(i16::MAX); } else { value = S::cast_from(i16::MIN); }
        } else if comptime!(code == 7) {
            if comptime!(op == 3) { value = S::cast_from(i8::MAX); } else { value = S::cast_from(i8::MIN); }
        } else if comptime!(code == 8) {
            if comptime!(op == 3) { value = S::cast_from(255u32); }
        } else {
            if comptime!(op == 3) { value = S::cast_from(f32::INFINITY); } else { value = S::cast_from(f32::NEG_INFINITY); }
        }
    }
    value
}

#[ruda]
fn combine<S: Numeric, F: Float>(a: S, ai: S, b: S, bi: S, #[comptime] op: u32, #[comptime] code: u32) -> (S, S) {
    let mut real = a; let mut imaginary = ai;
    if comptime!(code == 3) {
        let left = a != S::cast_from(0u32); let right = b != S::cast_from(0u32);
        if comptime!(op == 1 || op == 3) { real = S::cast_from(left && right); }
        else { real = S::cast_from(left || right); }
    } else if comptime!(op == 3 || op == 4) {
        let mut replace = b != b;
        if comptime!(op == 3) { replace = replace || b < a; }
        else { replace = replace || b > a; }
        if replace { real = b; }
    } else if comptime!(code >= 4 && code <= 8) {
        if comptime!(op == 1) { real = S::cast_from(u64::cast_from(a) * u64::cast_from(b)); }
        else { real = S::cast_from(u64::cast_from(a) + u64::cast_from(b)); }
    } else {
        let x = F::cast_from(a); let y = F::cast_from(b);
        if comptime!(op == 1) {
            if comptime!(code == 10 || code == 11) {
                let xi = F::cast_from(ai); let yi = F::cast_from(bi);
                real = S::cast_from(x * y - xi * yi);
                imaginary = S::cast_from(x * yi + xi * y);
            } else { real = S::cast_from(x * y); }
        } else {
            real = S::cast_from(x + y);
            if comptime!(code == 10 || code == 11) { imaginary = S::cast_from(F::cast_from(ai) + F::cast_from(bi)); }
        }
    }
    (real, imaginary)
}

#[ruda]
fn increment<S: Numeric, F: Float>(count: S, #[comptime] code: u32) -> S {
    let mut result = count;
    if comptime!(code >= 4 && code <= 8) { result = S::cast_from(u64::cast_from(count) + 1u64); }
    else { result = S::cast_from(F::cast_from(count) + F::new(1.0)); }
    result
}

#[ruda]
fn mean<S: Numeric, F: Float>(real: S, imaginary: S, count: S, #[comptime] code: u32) -> (S, S) {
    let mut divisor = count;
    if divisor == S::cast_from(0u32) { divisor = S::cast_from(1u32); }
    let mut result = real; let mut result_i = imaginary;
    if comptime!(code >= 4 && code <= 8) {
        let numerator = i64::cast_from(real); let denominator = i64::cast_from(divisor);
        let mut quotient = numerator / denominator;
        if numerator % denominator != 0i64 && ((numerator < 0i64) != (denominator < 0i64)) { quotient -= 1i64; }
        result = S::cast_from(quotient);
    } else if comptime!(code == 10 || code == 11) {
        let (r, i) = crate::foreach_unary::complex_divide(F::cast_from(real), F::cast_from(imaginary), F::cast_from(divisor), F::new(0.0));
        result = S::cast_from(r); result_i = S::cast_from(i);
    } else { result = S::cast_from(F::cast_from(real) / F::cast_from(divisor)); }
    (result, result_i)
}

#[ruda(launch)]
fn reduce_runs<S: Numeric + RudaElement, F: Float>(
    keys: &Tensor<i64>, order: &Tensor<u32>, source: &Tensor<S>, source_imaginary: &Tensor<S>,
    target: &mut Tensor<S>, target_imaginary: &mut Tensor<S>,
    #[comptime] op: u32, #[comptime] code: u32, #[comptime] include_self: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < order.len() {
        let key = keys[order[p] as usize];
        let mut head = p == 0usize;
        if p > 0usize { head = key != keys[order[p - 1usize] as usize]; }
        if head {
            let destination = key as usize;
            let mut real = identity::<S>(op, code); let mut imaginary = S::cast_from(0u32);
            let mut count = S::cast_from(0u32);
            if comptime!(include_self) {
                real = target[address(target, destination)]; count = S::cast_from(1u32);
                if comptime!(code == 10 || code == 11) { imaginary = target_imaginary[address(target_imaginary, destination)]; }
            }
            let mut cursor = p; let mut same = true;
            while cursor < order.len() && same {
                let from = order[cursor] as usize;
                if keys[from] != key { same = false; }
                else {
                    let value = source[address(source, from)]; let mut value_i = S::cast_from(0u32);
                    if comptime!(code == 10 || code == 11) { value_i = source_imaginary[address(source_imaginary, from)]; }
                    let (r, i) = combine::<S, F>(real, imaginary, value, value_i, op, code);
                    real = r; imaginary = i;
                    if comptime!(op == 2) { count = increment::<S, F>(count, code); }
                    cursor += 1usize;
                }
            }
            if comptime!(op == 2) {
                let (r, i) = mean::<S, F>(real, imaginary, count, code);
                real = r; imaginary = i;
            }
            target[address(target, destination)] = real;
            if comptime!(code == 10 || code == 11) { target_imaginary[address(target_imaginary, destination)] = imaginary; }
        }
    }
}

fn grid(work: usize) -> RudaCount {
    assert!(work > 0 && work <= u32::MAX as usize, "scatter reduction exceeds U32 indexing");
    let blocks = work.div_ceil(256); let x = blocks.min(131072);
    RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1)
}

fn scratch(c: &ComputeClient<CudaRuntime>, length: usize, dtype: u32) -> View {
    View::packed(c.empty(length.checked_mul(element_bytes(dtype)).expect("scatter reduction workspace overflow")), vec![length], dtype)
}

fn part(value: &View, imaginary: bool) -> View {
    if matches!(value.dtype, 10 | 11) {
        let result = scientific::component(value, imaginary);
        scientific::indexable(&result);
        result
    } else { View { handle: value.handle.clone(), shape: value.shape.clone(), strides: value.strides.clone(), len: value.len, dtype: value.dtype } }
}

pub(super) fn dispatch(xs: &[View], ints: &[i64]) {
    assert_eq!(xs.len(), 3); assert_eq!(ints.len(), 3);
    let source = &xs[0]; let indices = &xs[1]; let target = &xs[2];
    assert!(ints[0] >= 0 && (ints[0] as usize) < target.shape.len() && (0..=4).contains(&ints[1]) && matches!(ints[2], 0 | 1));
    let axis = ints[0] as usize; let op = ints[1] as u32; let include_self = ints[2] != 0;
    assert!(source.dtype <= 11 && matches!(indices.dtype, 4 | 5));
    assert_eq!(source.dtype, target.dtype); assert_eq!(source.shape, indices.shape); assert_eq!(source.shape.len(), target.shape.len());
    assert!(source.shape.iter().enumerate().all(|(dim, &size)| dim == axis || size <= target.shape[dim]));
    assert!(!(source.dtype == 3 && op == 2) && !(source.dtype >= 10 && op >= 3));
    if indices.len == 0 { return; }
    let c = client(); let block = RudaDim::new_1d(256); let point_grid = grid(indices.len);
    let keys = scratch(&c, indices.len, 4); let status = scratch(&c, 1, 5);
    unsafe { clear_status::launch::<CudaRuntime>(&c, RudaCount::Static(1, 1, 1), RudaDim::new_1d(1), status.arg()); }
    macro_rules! build_keys { ($i:ty) => { unsafe { offsets::launch::<$i, CudaRuntime>(&c, point_grid.clone(), block,
        indices.arg(), target.arg(), keys.arg(), status.arg(), axis); }}; }
    if indices.dtype == 4 { build_keys!(i64); } else { build_keys!(i32); }
    LAUNCHES.fetch_add(2, Ordering::Relaxed);
    sync(&c);
    let bytes = c.read_one(status.handle.clone()).expect("scatter reduction index status readback failed");
    DOWNLOAD.fetch_add(4, Ordering::Relaxed);
    assert_eq!(bytes.len(), 4);
    assert_eq!(u32::from_le_bytes(bytes[..4].try_into().unwrap()), 0, "scatter reduction index out of bounds");
    let order = scratch(&c, indices.len, 5);
    let (_sort_workspace, sorts) = unique_ops::stable_order(&c, &keys, &order, 1, true);
    let re = part(source, false); let im = part(source, true);
    let out_re = part(target, false); let out_im = part(target, true);
    macro_rules! run { ($s:ty, $f:ty) => { unsafe { reduce_runs::launch::<$s, $f, CudaRuntime>(&c, point_grid, block,
        keys.arg(), order.arg(), re.arg(), im.arg(), out_re.arg(), out_im.arg(), op, source.dtype, include_self); }}; }
    match source.dtype {
        0 | 10 => run!(f32, f32), 1 => run!(f16, f32), 2 => run!(bf16, f32), 3 | 8 => run!(u8, f32),
        4 => run!(i64, f32), 5 => run!(i32, f32), 6 => run!(i16, f32), 7 => run!(i8, f32), 9 | 11 => run!(f64, f64), _ => unreachable!(),
    }
    LAUNCHES.fetch_add(sorts + 1, Ordering::Relaxed);
    finish_dispatch(&c);
}
