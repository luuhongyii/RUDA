use super::*;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn block_id() -> usize {
    RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize
}

#[ruda(launch)]
fn batch_rows(indices: &Tensor<i64>, offsets: &mut Tensor<u32>, rows: usize) {
    let p = block_id() * 256usize + UNIT_POS as usize;
    if p < offsets.len() {
        let batch = i64::cast_from(p / rows);
        let row = i64::cast_from(p % rows);
        let mut lo = 0usize;
        let mut hi = indices.shape(1);
        while lo < hi {
            let mid = lo + (hi - lo) / 2usize;
            let b = indices[mid * indices.stride(1)];
            let r = indices[indices.stride(0) + mid * indices.stride(1)];
            if b < batch || (b == batch && r < row) {
                lo = mid + 1usize;
            } else {
                hi = mid;
            }
        }
        offsets[p] = u32::cast_from(lo);
    }
}

#[ruda(launch)]
fn batched_multiply<F: Float + RudaElement>(
    indices: &Tensor<i64>, offsets: &Tensor<u32>, ar: &Tensor<F>, ai: &Tensor<F>,
    br: &Tensor<F>, bi: &Tensor<F>, yr: &mut Tensor<F>, yi: &mut Tensor<F>,
    tiles: usize, #[comptime] complex: bool, #[comptime] vector: bool,
) {
    let block = block_id();
    let global_row = block / tiles;
    let batch = global_row / yr.shape(1);
    if batch >= yr.shape(0) { return; }
    let row = global_row % yr.shape(1);
    let tid = UNIT_POS as usize;
    let lane = tid % 32usize;
    let mut column = block % tiles * 32usize + lane;
    let mut entry_lane = tid / 32usize;
    let mut step = 8usize;
    if comptime!(vector) { column = 0usize; entry_lane = tid; step = 256usize; }
    let mut r = F::new(0.0);
    let mut i = F::new(0.0);
    if column < yr.shape(2) {
        let begin = offsets[global_row] as usize;
        let end = offsets[global_row + 1usize] as usize;
        let count = end - begin;
        let mut offset = entry_lane;
        let mut active = offset < count;
        while active {
            let entry = begin + offset;
            let k = indices[2usize * indices.stride(0) + entry * indices.stride(1)];
            if k >= 0i64 && k < i64::cast_from(br.shape(1)) {
                let bp = batch * br.stride(0) + k as usize * br.stride(1) + column * br.stride(2);
                let x = ar[entry * ar.stride(0)];
                let y = br[bp];
                if comptime!(complex) {
                    let u = ai[entry * ai.stride(0)];
                    let bip = batch * bi.stride(0) + k as usize * bi.stride(1) + column * bi.stride(2);
                    let v = bi[bip];
                    r += x * y - u * v;
                    i += x * v + u * y;
                } else {
                    r = fma(x, y, r);
                }
            }
            if count - offset <= step { active = false; } else { offset += step; }
        }
    }
    let mut write = tid < 32usize && column < yr.shape(2);
    if comptime!(vector) {
        r = crate::blas_accumulate::block_sum(r);
        if comptime!(complex) { i = crate::blas_accumulate::block_sum(i); }
        write = tid == 0usize;
    } else {
        let mut partial_r = SharedMemory::<F>::new(256usize);
        let mut partial_i = SharedMemory::<F>::new(256usize);
        partial_r[tid] = r;
        if comptime!(complex) { partial_i[tid] = i; }
        sync_ruda();
        if write {
            r = F::new(0.0);
            i = F::new(0.0);
            for warp in 0..8usize {
                r += partial_r[warp * 32usize + lane];
                if comptime!(complex) { i += partial_i[warp * 32usize + lane]; }
            }
        }
    }
    if write {
        yr[batch * yr.stride(0) + row * yr.stride(1) + column * yr.stride(2)] = r;
        if comptime!(complex) {
            yi[batch * yi.stride(0) + row * yi.stride(1) + column * yi.stride(2)] = i;
        }
    }
}

#[ruda(launch)]
fn hybrid_multiply<F: Float + RudaElement>(
    indices: &Tensor<i64>, ar: &Tensor<F>, ai: &Tensor<F>,
    br: &Tensor<F>, bi: &Tensor<F>, yr: &mut Tensor<F>, yi: &mut Tensor<F>,
    #[comptime] complex: bool,
) {
    let p = block_id() * 256usize + UNIT_POS as usize;
    if p < yr.len() {
        let entry = p % yr.shape(0);
        let column = p / yr.shape(0);
        let k = indices[indices.stride(0) + entry * indices.stride(1)];
        let mut r = F::new(0.0);
        let mut i = F::new(0.0);
        if k >= 0i64 && k < i64::cast_from(br.shape(0)) {
            let x = ar[entry * ar.stride(0)];
            let y = br[k as usize * br.stride(0) + column * br.stride(1)];
            if comptime!(complex) {
                let u = ai[entry * ai.stride(0)];
                let v = bi[k as usize * bi.stride(0) + column * bi.stride(1)];
                r = x * y - u * v;
                i = x * v + u * y;
            } else {
                r = x * y;
            }
        }
        yr[entry * yr.stride(0) + column * yr.stride(1)] = r;
        if comptime!(complex) { yi[entry * yi.stride(0) + column * yi.stride(1)] = i; }
    }
}

fn grid(blocks: usize) -> RudaCount {
    assert!(blocks > 0 && blocks <= u32::MAX as usize);
    let x = blocks.min(131072);
    RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1)
}

fn part(v: &View, imaginary: bool) -> View {
    if matches!(v.dtype, 10 | 11) {
        scientific::component(v, imaginary)
    } else {
        View { handle: v.handle.clone(), shape: v.shape.clone(), strides: v.strides.clone(), len: v.len, dtype: v.dtype }
    }
}

pub(super) fn dispatch(op: u32, xs: &[View]) {
    assert_eq!(xs.len(), 4);
    let (indices, values, dense, out) = (&xs[0], &xs[1], &xs[2], &xs[3]);
    assert!(op == 540 || op == 541);
    let rank = if op == 540 { 3 } else { 2 };
    assert_eq!(indices.dtype, 4);
    assert_eq!(indices.shape.len(), 2);
    assert_eq!(indices.shape[0], rank);
    assert_eq!(values.shape, vec![indices.shape[1]]);
    assert!(matches!(values.dtype, 0 | 9 | 10 | 11));
    for v in [dense, out] {
        assert_eq!(v.shape.len(), rank);
        assert_eq!(v.dtype, values.dtype);
    }
    assert_eq!(dense.shape[rank - 1], out.shape[rank - 1]);
    if op == 540 { assert_eq!(dense.shape[0], out.shape[0]); }
    else { assert_eq!(out.shape[0], values.len); }
    if out.len == 0 { return; }
    let c = client();
    let dim = RudaDim::new_1d(256);
    let parts: Vec<View> = [values, dense, out].iter().flat_map(|v| [part(v, false), part(v, true)]).collect();
    for v in &parts { scientific::indexable(v); }
    let complex = matches!(values.dtype, 10 | 11);
    if op == 540 {
        let rows = out.shape[0].checked_mul(out.shape[1]).expect("COO bmm row count overflow");
        let count = rows.checked_add(1).expect("COO bmm row offset count overflow");
        let offsets = View::packed(c.empty(count.checked_mul(4).expect("COO bmm row workspace overflow")), vec![count], 5);
        scientific::indexable(&offsets);
        let tiles = out.shape[2].div_ceil(32);
        let blocks = rows.checked_mul(tiles).expect("COO bmm grid overflow");
        unsafe {
            batch_rows::launch::<CudaRuntime>(&c, grid(count.div_ceil(256)), dim, indices.arg(), offsets.arg(), out.shape[1]);
        }
        macro_rules! run { ($f:ty) => { unsafe {
            batched_multiply::launch::<$f, CudaRuntime>(&c, grid(blocks), dim, indices.arg(), offsets.arg(),
                parts[0].arg(), parts[1].arg(), parts[2].arg(), parts[3].arg(), parts[4].arg(), parts[5].arg(),
                tiles, complex, out.shape[2] == 1);
        }}; }
        match values.dtype { 0 | 10 => run!(f32), 9 | 11 => run!(f64), _ => unreachable!() }
        LAUNCHES.fetch_add(2, Ordering::Relaxed);
    } else {
        macro_rules! run { ($f:ty) => { unsafe {
            hybrid_multiply::launch::<$f, CudaRuntime>(&c, grid(out.len.div_ceil(256)), dim, indices.arg(),
                parts[0].arg(), parts[1].arg(), parts[2].arg(), parts[3].arg(), parts[4].arg(), parts[5].arg(), complex);
        }}; }
        match values.dtype { 0 | 10 => run!(f32), 9 | 11 => run!(f64), _ => unreachable!() }
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
    }
    finish_dispatch(&c);
}
