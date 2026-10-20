use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn clear_status(status: &mut Tensor<u32>) {
    if UNIT_POS == 0u32 { status[0] = 0u32; }
}

#[ruda(launch)]
fn check_signed(indices: &Tensor<i64>, status: &mut Tensor<Atomic<u32>>, length: i64) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < indices.len() {
        let index = indices[address(indices, p)]; let limit = length;
        if index < -limit || index >= limit { status[0].store(1u32); }
    }
}

#[ruda]
fn wrapped(index: i64, length: usize, #[comptime] negative: bool) -> usize {
    let mut result = index;
    if comptime!(negative) { if result < 0i64 { result += i64::cast_from(length); } }
    usize::cast_from(result)
}

#[ruda(launch)]
fn assign_axis<S: Numeric + RudaElement, I: Int + RudaElement>(
    source: &Tensor<S>, source_imaginary: &Tensor<S>, indices: &Tensor<I>,
    target: &mut Tensor<S>, target_imaginary: &mut Tensor<S>, work: usize,
    #[comptime] axis: usize, #[comptime] mode: u32, #[comptime] complex: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < work {
        let mut remaining = p; let mut dim = target.rank(); let mut to = 0usize; let mut at = p;
        while dim > 0usize {
            dim -= 1usize;
            let mut size = 0usize;
            if comptime!(mode == 2) {
                size = target.shape(dim);
                if dim == axis { size = indices.len(); }
            } else { size = source.shape(dim); }
            let coordinate = remaining % size; remaining /= size;
            if dim == axis {
                if comptime!(mode != 0) { at = coordinate; }
            } else { to += coordinate * target.stride(dim); }
        }
        let index = wrapped(i64::cast_from(indices[address(indices, at)]), target.shape(axis), mode == 2);
        to += index * target.stride(axis);
        let mut from = 0usize;
        if comptime!(mode != 2) { from = address(source, p); }
        target[to] = source[from];
        if comptime!(complex) { target_imaginary[to] = source_imaginary[from]; }
    }
}

#[ruda(launch)]
fn take<S: Numeric + RudaElement>(
    source: &Tensor<S>, source_imaginary: &Tensor<S>, indices: &Tensor<i64>,
    output: &mut Tensor<S>, output_imaginary: &mut Tensor<S>,
    #[comptime] negative: bool, #[comptime] complex: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < output.len() {
        let index = wrapped(indices[address(indices, p)], source.len(), negative);
        let from = address(source, index); let to = address(output, p);
        output[to] = source[from];
        if comptime!(complex) { output_imaginary[to] = source_imaginary[from]; }
    }
}

#[ruda(launch)]
fn masked_take<S: Numeric + RudaElement>(
    source: &Tensor<S>, source_imaginary: &Tensor<S>, indices: &Tensor<i64>, mask: &Tensor<u8>,
    fill: &Tensor<S>, fill_imaginary: &Tensor<S>, output: &mut Tensor<S>, output_imaginary: &mut Tensor<S>,
    #[comptime] complex: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < output.len() {
        let mut value = fill[0usize]; let mut value_i = S::cast_from(0u32);
        if comptime!(complex) { value_i = fill_imaginary[0usize]; }
        if source.len() > 0usize && mask[address(mask, p)] != 0u8 {
            let from = address(source, usize::cast_from(indices[address(indices, p)]));
            value = source[from];
            if comptime!(complex) { value_i = source_imaginary[from]; }
        }
        let to = address(output, p); output[to] = value;
        if comptime!(complex) { output_imaginary[to] = value_i; }
    }
}

#[ruda(launch)]
fn put<S: Numeric + RudaElement>(
    source: &Tensor<S>, source_imaginary: &Tensor<S>, indices: &Tensor<i64>,
    target: &mut Tensor<S>, target_imaginary: &mut Tensor<S>,
    #[comptime] negative: bool, #[comptime] complex: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < indices.len() {
        let index = wrapped(indices[address(indices, p)], target.len(), negative);
        let from = address(source, p); let to = address(target, index);
        target[to] = source[from];
        if comptime!(complex) { target_imaginary[to] = source_imaginary[from]; }
    }
}

#[ruda(launch)]
fn normalize(indices: &Tensor<i64>, keys: &mut Tensor<i64>, length: usize) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < indices.len() { keys[p] = i64::cast_from(wrapped(indices[address(indices, p)], length, true)); }
}

#[ruda]
fn add<S: Numeric, F: Float>(a: S, b: S, #[comptime] code: u32) -> S {
    let mut result = a;
    if comptime!(code == 3) { result = S::cast_from(a != S::cast_from(0u32) || b != S::cast_from(0u32)); }
    else if comptime!(code >= 4 && code <= 8) { result = S::cast_from(u64::cast_from(a) + u64::cast_from(b)); }
    else { result = S::cast_from(F::cast_from(a) + F::cast_from(b)); }
    result
}

#[ruda(launch)]
fn add_runs<S: Numeric + RudaElement, F: Float>(
    source: &Tensor<S>, source_imaginary: &Tensor<S>, keys: &Tensor<i64>, order: &Tensor<u32>,
    target: &mut Tensor<S>, target_imaginary: &mut Tensor<S>, mask: &Tensor<u8>,
    #[comptime] code: u32, #[comptime] masked: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < order.len() {
        let first = order[p] as usize; let key = keys[address(keys, first)];
        let mut head = p == 0usize;
        if p > 0usize { head = key != keys[address(keys, order[p - 1usize] as usize)]; }
        if head {
            let to = address(target, usize::cast_from(key));
            let mut value = target[to]; let mut value_i = S::cast_from(0u32);
            if comptime!(code == 10 || code == 11) { value_i = target_imaginary[to]; }
            let mut next = p; let mut active = true;
            while next < order.len() && active {
                let position = order[next] as usize;
                if keys[address(keys, position)] == key {
                    let mut selected = true;
                    if comptime!(masked) { selected = mask[address(mask, position)] != 0u8; }
                    let mut term = S::cast_from(0u32); let mut term_i = S::cast_from(0u32);
                    if selected {
                        let from = address(source, position); term = source[from];
                        if comptime!(code == 10 || code == 11) { term_i = source_imaginary[from]; }
                    }
                    value = add::<S, F>(value, term, code);
                    if comptime!(code == 10 || code == 11) { value_i = add::<S, F>(value_i, term_i, code); }
                    next += 1usize;
                } else { active = false; }
            }
            target[to] = value;
            if comptime!(code == 10 || code == 11) { target_imaginary[to] = value_i; }
        }
    }
}

fn grid(work: usize) -> RudaCount {
    assert!(work > 0 && work <= u32::MAX as usize, "indexed write exceeds U32 indexing");
    let blocks = work.div_ceil(256); let x = blocks.min(131072);
    RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1)
}

fn scratch(c: &ComputeClient<CudaRuntime>, length: usize, dtype: u32) -> View {
    assert!(length <= u32::MAX as usize, "indexed write workspace exceeds U32 indexing");
    View::packed(c.empty(length.checked_mul(element_bytes(dtype)).expect("indexed write workspace overflow")), vec![length], dtype)
}

pub(super) fn validate(c: &ComputeClient<CudaRuntime>, indices: &View, length: usize, negative: bool) {
    if !negative { embedding_ops::validate_indices(c, indices, length); return; }
    assert_eq!(indices.dtype, 4);
    if indices.len == 0 { return; }
    if std::env::var("RUDA_TORCH_INDEX_CHECK").as_deref() == Ok("trusted") {
        TRUSTED_INDEX_CALLS.fetch_add(1, Ordering::Relaxed); return;
    }
    let status = scratch(c, 1, 5);
    unsafe {
        clear_status::launch::<CudaRuntime>(c, RudaCount::Static(1, 1, 1), RudaDim::new_1d(1), status.arg());
        check_signed::launch::<CudaRuntime>(c, grid(indices.len), RudaDim::new_1d(256), indices.arg(), status.arg(),
            i64::try_from(length).expect("index dimension exceeds int64"));
    }
    LAUNCHES.fetch_add(2, Ordering::Relaxed); sync(c);
    let bytes = c.read_one(status.handle.clone()).expect("indexed write status readback failed");
    DOWNLOAD.fetch_add(4, Ordering::Relaxed); assert_eq!(bytes.len(), 4);
    assert_eq!(u32::from_le_bytes(bytes[..4].try_into().unwrap()), 0, "RUDA index out of bounds");
}

fn part(value: &View, imaginary: bool) -> View {
    if matches!(value.dtype, 10 | 11 | 14) {
        let result = scientific::component(value, imaginary);
        scientific::indexable(&result); result
    } else { View { handle: value.handle.clone(), shape: value.shape.clone(), strides: value.strides.clone(), len: value.len, dtype: value.dtype } }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64]) {
    assert!((472..=474).contains(&op) || matches!(op, 476 | 477));
    assert_eq!(xs.len(), if op == 476 { 5 } else if op == 477 { 4 } else { 3 });
    let source = &xs[0]; let indices = &xs[1]; let target = &xs[2];
    assert!(source.dtype <= 14); assert_eq!(source.dtype, target.dtype);
    assert!(matches!(indices.dtype, 4 | 5));
    let c = client(); let block = RudaDim::new_1d(256);
    let re = part(source, false); let im = part(source, true); let out_re = part(target, false); let out_im = part(target, true);
    let complex = matches!(source.dtype, 10 | 11 | 14);
    if op == 472 {
        assert_eq!(ints.len(), 2); assert!(ints[0] >= 0 && (0..=2).contains(&ints[1]));
        let axis = ints[0] as usize; let mode = ints[1] as u32;
        assert!(axis < target.shape.len());
        if mode != 0 { assert_eq!(indices.dtype, 4); assert_eq!(indices.shape.len(), 1); }
        let work;
        if mode == 2 {
            assert_eq!(source.len, 1);
            let rows = target.shape.iter().enumerate().filter(|(dim, _)| *dim != axis)
                .try_fold(1usize, |n, (_, &size)| n.checked_mul(size)).expect("indexed fill shape overflow");
            work = rows.checked_mul(indices.len).expect("indexed fill shape overflow");
        } else {
            assert_eq!(source.shape.len(), target.shape.len());
            if mode == 0 {
                assert_eq!(source.shape, indices.shape);
                assert!(source.shape.iter().enumerate().all(|(dim, &size)| dim == axis || size <= target.shape[dim]));
            } else {
                assert_eq!(source.shape[axis], indices.len);
                assert!(source.shape.iter().enumerate().all(|(dim, &size)| dim == axis || size == target.shape[dim]));
            }
            work = source.len;
        }
        if work == 0 { return; }
        let launch_grid = grid(work);
        validate(&c, indices, target.shape[axis], mode == 2);
        macro_rules! launch { ($s:ty, $i:ty) => { unsafe { assign_axis::launch::<$s, $i, CudaRuntime>(&c, launch_grid, block,
            re.arg(), im.arg(), indices.arg(), out_re.arg(), out_im.arg(), work, axis, mode, complex); }}; }
        macro_rules! data { ($s:ty) => { if indices.dtype == 4 { launch!($s, i64); } else { launch!($s, i32); } }; }
        match re.dtype {
            0 => data!(f32), 1 => data!(f16), 2 => data!(bf16), 3 | 8 | 12 | 13 => data!(u8),
            4 => data!(i64), 5 => data!(i32), 6 => data!(i16), 7 => data!(i8), 9 => data!(f64), _ => unreachable!(),
        }
    } else if op == 473 {
        assert_eq!(ints.len(), 1); assert!(matches!(ints[0], 0 | 1)); assert_eq!(indices.dtype, 4);
        assert_eq!(target.shape, indices.shape);
        if target.len == 0 { return; }
        let negative = ints[0] != 0;
        validate(&c, indices, source.len, negative);
        macro_rules! launch { ($s:ty) => { unsafe { take::launch::<$s, CudaRuntime>(&c, grid(target.len), block,
            re.arg(), im.arg(), indices.arg(), out_re.arg(), out_im.arg(), negative, complex); }}; }
        match re.dtype {
            0 => launch!(f32), 1 => launch!(f16), 2 => launch!(bf16), 3 | 8 | 12 | 13 => launch!(u8),
            4 => launch!(i64), 5 => launch!(i32), 6 => launch!(i16), 7 => launch!(i8), 9 => launch!(f64), _ => unreachable!(),
        }
    } else if op == 476 {
        assert!(ints.is_empty()); assert_eq!(indices.dtype, 4); assert_eq!(target.shape, indices.shape);
        let mask = &xs[3]; let fill = &xs[4];
        assert_eq!(mask.dtype, 3); assert_eq!(mask.shape, target.shape); assert_eq!(fill.dtype, source.dtype); assert_eq!(fill.len, 1);
        if target.len == 0 { return; }
        if source.len > 0 { validate(&c, indices, source.len, false); }
        let fill_re = part(fill, false); let fill_im = part(fill, true);
        macro_rules! launch { ($s:ty) => { unsafe { masked_take::launch::<$s, CudaRuntime>(&c, grid(target.len), block,
            re.arg(), im.arg(), indices.arg(), mask.arg(), fill_re.arg(), fill_im.arg(), out_re.arg(), out_im.arg(), complex); }}; }
        match re.dtype {
            0 => launch!(f32), 1 => launch!(f16), 2 => launch!(bf16), 3 | 8 | 12 | 13 => launch!(u8),
            4 => launch!(i64), 5 => launch!(i32), 6 => launch!(i16), 7 => launch!(i8), 9 => launch!(f64), _ => unreachable!(),
        }
    } else {
        assert_eq!(ints.len(), 2); assert!(ints.iter().all(|x| matches!(*x, 0 | 1))); assert_eq!(indices.dtype, 4);
        assert_eq!(source.len, indices.len);
        let accumulate = ints[0] != 0; let negative = ints[1] != 0;
        let masked = op == 477; let mask = if masked { &xs[3] } else { indices };
        if masked {
            assert!(accumulate && !negative); assert_eq!(mask.dtype, 3); assert_eq!(mask.shape, source.shape);
        }
        assert!(!accumulate || source.dtype <= 11, "indexed accumulation does not dispatch float8 or complex32");
        if indices.len == 0 { return; }
        validate(&c, indices, target.len, negative);
        if !accumulate {
            macro_rules! launch { ($s:ty) => { unsafe { put::launch::<$s, CudaRuntime>(&c, grid(indices.len), block,
                re.arg(), im.arg(), indices.arg(), out_re.arg(), out_im.arg(), negative, complex); }}; }
            match re.dtype {
                0 => launch!(f32), 1 => launch!(f16), 2 => launch!(bf16), 3 | 8 | 12 | 13 => launch!(u8),
                4 => launch!(i64), 5 => launch!(i32), 6 => launch!(i16), 7 => launch!(i8), 9 => launch!(f64), _ => unreachable!(),
            }
        } else {
            let normalized = if negative {
                let keys = scratch(&c, indices.len, 4);
                unsafe { normalize::launch::<CudaRuntime>(&c, grid(indices.len), block, indices.arg(), keys.arg(), target.len); }
                LAUNCHES.fetch_add(1, Ordering::Relaxed); Some(keys)
            } else { None };
            let keys = normalized.as_ref().unwrap_or(indices);
            let order = scratch(&c, indices.len, 5);
            let (_sort_workspace, sorts) = unique_ops::stable_order(&c, keys, &order, 1, true);
            macro_rules! launch { ($s:ty, $f:ty) => { unsafe { add_runs::launch::<$s, $f, CudaRuntime>(&c, grid(indices.len), block,
                re.arg(), im.arg(), keys.arg(), order.arg(), out_re.arg(), out_im.arg(), mask.arg(), source.dtype, masked); }}; }
            match source.dtype {
                0 | 10 => launch!(f32, f32), 1 => launch!(f16, f32), 2 => launch!(bf16, f32), 3 | 8 => launch!(u8, f32),
                4 => launch!(i64, f32), 5 => launch!(i32, f32), 6 => launch!(i16, f32), 7 => launch!(i8, f32), 9 | 11 => launch!(f64, f64), _ => unreachable!(),
            }
            LAUNCHES.fetch_add(sorts + 1, Ordering::Relaxed); finish_dispatch(&c); return;
        }
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed); finish_dispatch(&c);
}
