use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn transpose_rows<I: Numeric + RudaElement>(pointers: &Tensor<I>, order: &Tensor<i64>, rows: &mut Tensor<I>) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < rows.len() {
        let entry = order[address(order, p)];
        let mut begin = 0usize; let mut end = pointers.len() - 1usize;
        while begin < end {
            let middle = begin + (end - begin) / 2usize;
            if i64::cast_from(pointers[address(pointers, middle + 1usize)]) <= entry { begin = middle + 1usize; }
            else { end = middle; }
        }
        rows[address(rows, p)] = I::cast_from(begin);
    }
}

#[ruda]
fn reduce<F: Float>(value: F, #[comptime] vector: bool) -> F {
    let mut total = F::new(0.0);
    if comptime!(vector) {
        total = crate::blas_accumulate::block_sum(value);
        sync_ruda();
    } else {
        let tid = UNIT_POS as usize; let mut partial = SharedMemory::<F>::new(256usize);
        partial[tid] = value; sync_ruda();
        if tid < 32usize { for warp in 0..8usize { total += partial[warp * 32usize + tid]; } }
        sync_ruda();
    }
    total
}

#[ruda]
fn missing_block_diagonal<I: Numeric>(pointers: &Tensor<I>, indices: &Tensor<I>) -> u32 {
    let tid = UNIT_POS as usize; let rows = pointers.len() - 1usize;
    let mut row = tid; let mut active = row < rows; let mut missing = 0u32;
    while active {
        let start = usize::cast_from(pointers[address(pointers, row)]);
        let end = usize::cast_from(pointers[address(pointers, row + 1usize)]);
        let mut found = false;
        for entry in start..end { if usize::cast_from(indices[address(indices, entry)]) == row { found = true; } }
        if !found { missing = 1u32; }
        if rows - row <= 256usize { active = false; } else { row += 256usize; }
    }
    let mut flags = SharedMemory::<u32>::new(256usize); flags[tid] = missing; sync_ruda();
    let mut width = 128usize;
    while width > 0usize {
        if tid < width { flags[tid] = flags[tid] | flags[tid + width]; }
        sync_ruda(); width /= 2usize;
    }
    let result = flags[0]; sync_ruda(); result
}

#[ruda(launch)]
fn solve<I: Numeric + RudaElement, F: Float + RudaElement>(
    pointers: &Tensor<I>, indices: &Tensor<I>, ar: &Tensor<F>, ai: &Tensor<F>,
    br: &Tensor<F>, bi: &Tensor<F>, xr: &mut Tensor<F>, xi: &mut Tensor<F>, order: &Tensor<i64>,
    #[comptime] block_size: usize, #[comptime] upper: bool, #[comptime] unit: bool,
    #[comptime] transpose: bool, #[comptime] complex: bool, #[comptime] vector: bool,
    #[comptime] code: u32, #[comptime] fraction: u32, #[comptime] exponent: u32,
) {
    let tile = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    let tid = UNIT_POS as usize; let mut column = tile * 32usize + tid % 32usize;
    let mut entry_lane = tid / 32usize; let mut entry_step = 8usize;
    let mut write = tid < 32usize && column < xr.shape(1);
    if comptime!(vector) { column = 0usize; entry_lane = tid; entry_step = 256usize; write = tid == 0usize; }
    let zero = F::new(0.0);
    if comptime!(block_size > 1 && !unit) {
        if missing_block_diagonal(pointers, indices) != 0u32 {
            if write {
                for row in 0..xr.shape(0) {
                    xr[row * xr.stride(0) + column * xr.stride(1)] = F::cast_from(f32::NAN);
                    if comptime!(complex) { xi[row * xi.stride(0) + column * xi.stride(1)] = zero; }
                }
            }
            return;
        }
    }
    for step in 0..xr.shape(0) {
        let mut row = step; if comptime!(upper) { row = xr.shape(0) - 1usize - step; }
        let compressed_row = row / block_size; let inner_row = row % block_size;
        let start = usize::cast_from(pointers[address(pointers, compressed_row)]) * block_size;
        let end = usize::cast_from(pointers[address(pointers, compressed_row + 1usize)]) * block_size;
        let count = end - start;
        let mut real = zero; let mut imag = zero;
        if column < xr.shape(1) {
            let mut offset = entry_lane; let mut active = offset < count;
            while active {
                let term = start + offset; let entry = term / block_size; let inner_col = term % block_size;
                let k = usize::cast_from(indices[address(indices, entry)]) * block_size + inner_col;
                let mut prior = k < row; if comptime!(upper) { prior = k > row && k < xr.shape(0); }
                if prior {
                    let mut source = entry; let mut vr = inner_row; let mut vc = inner_col;
                    if comptime!(transpose) { source = usize::cast_from(order[address(order, entry)]); vr = inner_col; vc = inner_row; }
                    let a = ar[source * ar.stride(0) + vr * ar.stride(1) + vc * ar.stride(2)];
                    let mut b = zero;
                    if comptime!(complex) { b = ai[source * ai.stride(0) + vr * ai.stride(1) + vc * ai.stride(2)]; }
                    let x = xr[k * xr.stride(0) + column * xr.stride(1)];
                    if comptime!(complex) {
                        let y = xi[k * xi.stride(0) + column * xi.stride(1)];
                        real += a * x - b * y; imag += a * y + b * x;
                    } else { real = fma(a, x, real); }
                }
                if count - offset <= entry_step { active = false; } else { offset += entry_step; }
            }
        }
        real = reduce(real, vector);
        if comptime!(complex) { imag = reduce(imag, vector); }
        if write {
            let mut x = br[row * br.stride(0) + column * br.stride(1)] - real; let mut y = zero;
            if comptime!(complex) { y = bi[row * bi.stride(0) + column * bi.stride(1)] - imag; }
            if comptime!(!unit) {
                let mut diagonal_r = zero; let mut diagonal_i = zero;
                for entry in start / block_size..end / block_size {
                    if usize::cast_from(indices[address(indices, entry)]) == compressed_row {
                        let mut source = entry;
                        if comptime!(transpose) { source = usize::cast_from(order[address(order, entry)]); }
                        diagonal_r = ar[source * ar.stride(0) + inner_row * ar.stride(1) + inner_row * ar.stride(2)];
                        if comptime!(complex) {
                            diagonal_i = ai[source * ai.stride(0) + inner_row * ai.stride(1) + inner_row * ai.stride(2)];
                        }
                    }
                }
                if comptime!(complex) {
                    let (r, i) = crate::math_extra::complex_divide(x, y, diagonal_r, diagonal_i, code, fraction, exponent);
                    x = r; y = i;
                } else { x /= diagonal_r; }
            }
            xr[row * xr.stride(0) + column * xr.stride(1)] = x;
            if comptime!(complex) { xi[row * xi.stride(0) + column * xi.stride(1)] = y; }
        }
        sync_ruda();
    }
}

fn grid(blocks: usize) -> RudaCount {
    assert!(blocks > 0 && blocks <= u32::MAX as usize);
    let x = blocks.min(131072); RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1)
}

fn part(v: &View, imaginary: bool) -> View {
    if matches!(v.dtype, 10 | 11) { scientific::component(v, imaginary) }
    else { View { handle: v.handle.clone(), shape: v.shape.clone(), strides: v.strides.clone(), len: v.len, dtype: v.dtype } }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64]) {
    assert_eq!(xs.len(), if op == 536 { 6 } else { 3 });
    let index_dtype = xs[0].dtype; assert!(matches!(index_dtype, 4 | 5));
    assert_eq!(xs[0].shape.len(), 1); assert!(xs[0].len > 0);
    let c = client(); let dim = RudaDim::new_1d(256);
    if op == 537 {
        assert!(ints.is_empty()); assert_eq!(xs[1].dtype, 4); assert_eq!(xs[1].shape.len(), 1);
        assert_eq!(xs[2].dtype, index_dtype); assert_eq!(xs[2].shape, xs[1].shape);
        if xs[2].len == 0 { return; }
        macro_rules! run { ($i:ty) => { unsafe {
            transpose_rows::launch::<$i, CudaRuntime>(&c, grid(xs[2].len.div_ceil(256)), dim, xs[0].arg(), xs[1].arg(), xs[2].arg());
        }}; }
        if index_dtype == 4 { run!(i64); } else { run!(i32); }
    } else {
        assert_eq!(op, 536); assert_eq!(ints.len(), 3); assert!(ints.iter().all(|&x| x == 0 || x == 1));
        let dtype = xs[2].dtype; assert!(matches!(dtype, 0 | 9 | 10 | 11));
        assert_eq!(xs[1].dtype, index_dtype); assert_eq!(xs[1].shape.len(), 1);
        assert_eq!(xs[2].shape.len(), 3); assert_eq!(xs[2].shape[0], xs[1].len);
        let block_size = xs[2].shape[1]; assert!(block_size > 0); assert_eq!(xs[2].shape[2], block_size);
        for v in &xs[3..5] { assert_eq!(v.dtype, dtype); assert_eq!(v.shape.len(), 2); }
        assert_eq!(xs[3].shape, xs[4].shape); assert_eq!(xs[4].shape[0], (xs[0].len - 1) * block_size);
        assert_eq!(xs[5].dtype, 4); assert_eq!(xs[5].shape.len(), 1);
        assert_eq!(xs[5].len, if ints[2] != 0 { xs[1].len } else { 0 });
        if xs[4].len == 0 { return; }
        let parts: Vec<View> = xs[2..5].iter().flat_map(|v| [part(v, false), part(v, true)]).collect();
        for v in &parts { scientific::indexable(v); }
        let blocks = xs[4].shape[1].div_ceil(32);
        macro_rules! run { ($i:ty, $f:ty, $code:expr, $fraction:expr, $exponent:expr) => { unsafe {
            solve::launch::<$i, $f, CudaRuntime>(&c, grid(blocks), dim, xs[0].arg(), xs[1].arg(),
                parts[0].arg(), parts[1].arg(), parts[2].arg(), parts[3].arg(), parts[4].arg(), parts[5].arg(), xs[5].arg(),
                block_size, ints[0] != 0, ints[1] != 0, ints[2] != 0, matches!(dtype, 10 | 11), xs[4].shape[1] == 1,
                $code, $fraction, $exponent);
        }}; }
        macro_rules! typed { ($f:ty, $code:expr, $fraction:expr, $exponent:expr) => {
            if index_dtype == 4 { run!(i64, $f, $code, $fraction, $exponent); }
            else { run!(i32, $f, $code, $fraction, $exponent); }
        }; }
        match dtype { 0 | 10 => typed!(f32, 0, 23, 8), 9 | 11 => typed!(f64, 9, 52, 11), _ => unreachable!() }
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed); finish_dispatch(&c);
}
