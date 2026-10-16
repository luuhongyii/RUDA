use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
pub(super) fn block_sum<F: Float>(value: F) -> F {
    let tid = UNIT_POS as usize;
    let mut total = value;
    let mut offset = 16u32;
    while offset > 0u32 {
        total += plane_shuffle_down(total, offset);
        offset /= 2u32;
    }
    let mut partial = SharedMemory::<F>::new(8usize);
    sync_ruda();
    if tid % 32usize == 0 { partial[tid / 32usize] = total; }
    sync_ruda();
    total = F::new(0.0);
    if tid < 8usize { total = partial[tid]; }
    if tid < 32usize {
        offset = 16u32;
        while offset > 0u32 {
            total += plane_shuffle_down(total, offset);
            offset /= 2u32;
        }
    }
    total
}

#[ruda(launch)]
fn gemv<S: Float + RudaElement, F: Float>(
    bias: &Tensor<S>, matrix: &Tensor<S>, vector: &Tensor<S>, out: &mut Tensor<S>,
    alpha: F, beta: F, #[comptime] zero_beta: bool,
) {
    let row = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    if row >= out.shape(0) { return; }
    let tid = UNIT_POS as usize;
    let inner = matrix.shape(1);
    let mut sum = F::new(0.0);
    if alpha != F::new(0.0) && inner != 0usize {
        let steps = inner / 256usize + usize::cast_from(inner % 256usize != 0usize);
        for step in 0..steps {
            let k = step * 256usize + tid;
            if k < inner {
                sum = fma(F::cast_from(matrix[row * matrix.stride(0) + k * matrix.stride(1)]),
                    F::cast_from(vector[k * vector.stride(0)]), sum);
            }
        }
    }
    let sum = block_sum(sum);
    if tid == 0usize {
        let mut value = F::new(0.0);
        if inner == 0usize {
            if comptime!(!zero_beta) { value = beta * F::cast_from(bias[row * bias.stride(0)]); }
        } else {
            value = alpha * sum;
            if beta != F::new(0.0) { value += beta * F::cast_from(bias[row * bias.stride(0)]); }
        }
        out[row * out.stride(0)] = S::cast_from(value);
    }
}

#[ruda(launch)]
fn complex_gemv<F: Float + RudaElement>(
    cr: &Tensor<F>, ci: &Tensor<F>, ar: &Tensor<F>, ai: &Tensor<F>, br: &Tensor<F>, bi: &Tensor<F>,
    yr: &mut Tensor<F>, yi: &mut Tensor<F>, alpha_r: F, alpha_i: F, beta_r: F, beta_i: F,
    #[comptime] zero_beta: bool,
) {
    let row = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    if row >= yr.shape(0) { return; }
    let tid = UNIT_POS as usize;
    let inner = ar.shape(1);
    let mut sr = F::new(0.0); let mut si = F::new(0.0);
    if (alpha_r != F::new(0.0) || alpha_i != F::new(0.0)) && inner != 0usize {
        let steps = inner / 256usize + usize::cast_from(inner % 256usize != 0usize);
        for step in 0..steps {
            let k = step * 256usize + tid;
            if k < inner {
                let x = ar[row * ar.stride(0) + k * ar.stride(1)];
                let u = ai[row * ai.stride(0) + k * ai.stride(1)];
                let y = br[k * br.stride(0)]; let v = bi[k * bi.stride(0)];
                sr += x * y - u * v; si += x * v + u * y;
            }
        }
    }
    let sr = block_sum(sr); let si = block_sum(si);
    if tid == 0usize {
        let mut r = F::new(0.0); let mut i = F::new(0.0);
        if inner != 0usize { r = alpha_r * sr - alpha_i * si; i = alpha_r * si + alpha_i * sr; }
        let mut read_bias = beta_r != F::new(0.0) || beta_i != F::new(0.0);
        if inner == 0usize { read_bias = comptime!(!zero_beta); }
        if read_bias {
            let x = cr[row * cr.stride(0)]; let y = ci[row * ci.stride(0)];
            r += beta_r * x - beta_i * y; i += beta_r * y + beta_i * x;
        }
        yr[row * yr.stride(0)] = r; yi[row * yi.stride(0)] = i;
    }
}

#[ruda(launch)]
fn batched<S: Float + RudaElement, O: Float + RudaElement, F: Float>(
    bias: &Tensor<O>, left: &Tensor<S>, right: &Tensor<S>, out: &mut Tensor<O>, alpha: F, beta: F,
    row_tiles: usize, column_tiles: usize, inner_tiles: usize,
    #[comptime] reduce_batch: bool, #[comptime] zero_beta: bool, #[comptime] activation: u32,
) {
    let block = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    let tid = UNIT_POS as usize; let tx = tid % 16usize; let ty = tid / 16usize;
    let tiles = row_tiles * column_tiles;
    let batch = block / tiles; let tile = block % tiles;
    if comptime!(reduce_batch) { if batch != 0usize { return; } }
    else { if batch >= left.shape(0) { return; } }
    let row = tile / column_tiles * 16usize + ty; let col = tile % column_tiles * 16usize + tx;
    let rows = left.shape(1); let inner = left.shape(2); let columns = right.shape(2);
    let valid = row < rows && col < columns;
    let p = (batch * rows + row) * columns + col;
    let mut batches = 1usize;
    if comptime!(reduce_batch) { batches = left.shape(0); }
    let mut value = F::new(0.0);
    let mut read_bias = beta != F::new(0.0);
    if inner == 0usize || batches == 0usize { read_bias = comptime!(!zero_beta); }
    if valid && read_bias { value = beta * F::cast_from(bias[address(bias, p)]); }
    if inner != 0usize && batches != 0usize {
        let mut a_tile = SharedMemory::<F>::new(256usize);
        let mut b_tile = SharedMemory::<F>::new(256usize);
        for sample in 0..batches {
            let mut source_batch = batch;
            if comptime!(reduce_batch) { source_batch = sample; }
            let mut sum = F::new(0.0);
            if alpha != F::new(0.0) {
                for tile_k in 0..inner_tiles {
                    let ak = tile_k * 16usize + tx; let bk = tile_k * 16usize + ty;
                    let mut av = F::new(0.0); let mut bv = F::new(0.0);
                    if row < rows && ak < inner {
                        av = F::cast_from(left[source_batch * left.stride(0) + row * left.stride(1) + ak * left.stride(2)]);
                    }
                    if bk < inner && col < columns {
                        bv = F::cast_from(right[source_batch * right.stride(0) + bk * right.stride(1) + col * right.stride(2)]);
                    }
                    a_tile[tid] = av; b_tile[tid] = bv;
                    sync_ruda();
                    for k in 0..16usize {
                        if tile_k * 16usize + k < inner { sum = fma(a_tile[ty * 16usize + k], b_tile[k * 16usize + tx], sum); }
                    }
                    sync_ruda();
                }
            }
            value = F::cast_from(O::cast_from(alpha * sum + value));
        }
    }
    if valid {
        if comptime!(activation != 0) {
            if inner != 0usize {
                if comptime!(activation == 1) {
                    if value == value { value = value.max(F::new(0.0)); }
                } else {
                    let beta = F::cast_from(0.797884560802865355879892119868764f64);
                    let kappa = F::cast_from(0.044715f64);
                    let cubic = value * value * value;
                    value = F::new(0.5) * value * (F::new(1.0) + (beta * (value + kappa * cubic)).tanh());
                }
            }
        }
        out[address(out, p)] = O::cast_from(value);
    }
}

#[ruda(launch)]
fn complex_batched<F: Float + RudaElement>(
    cr: &Tensor<F>, ci: &Tensor<F>, ar: &Tensor<F>, ai: &Tensor<F>, br: &Tensor<F>, bi: &Tensor<F>,
    yr: &mut Tensor<F>, yi: &mut Tensor<F>, alpha_r: F, alpha_i: F, beta_r: F, beta_i: F,
    row_tiles: usize, column_tiles: usize, inner_tiles: usize,
    #[comptime] reduce_batch: bool, #[comptime] zero_beta: bool,
) {
    let block = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    let tid = UNIT_POS as usize; let tx = tid % 16usize; let ty = tid / 16usize;
    let tiles = row_tiles * column_tiles;
    let batch = block / tiles; let tile = block % tiles;
    if comptime!(reduce_batch) { if batch != 0usize { return; } }
    else { if batch >= ar.shape(0) { return; } }
    let row = tile / column_tiles * 16usize + ty; let col = tile % column_tiles * 16usize + tx;
    let rows = ar.shape(1); let inner = ar.shape(2); let columns = br.shape(2);
    let valid = row < rows && col < columns;
    let p = (batch * rows + row) * columns + col;
    let mut batches = 1usize;
    if comptime!(reduce_batch) { batches = ar.shape(0); }
    let mut real = F::new(0.0); let mut imag = F::new(0.0);
    let mut read_bias = beta_r != F::new(0.0) || beta_i != F::new(0.0);
    if inner == 0usize || batches == 0usize { read_bias = comptime!(!zero_beta); }
    if valid && read_bias {
        let x = cr[address(cr, p)]; let y = ci[address(ci, p)];
        real = beta_r * x - beta_i * y; imag = beta_r * y + beta_i * x;
    }
    if inner != 0usize && batches != 0usize {
        let mut tile_ar = SharedMemory::<F>::new(256usize); let mut tile_ai = SharedMemory::<F>::new(256usize);
        let mut tile_br = SharedMemory::<F>::new(256usize); let mut tile_bi = SharedMemory::<F>::new(256usize);
        for sample in 0..batches {
            let mut source_batch = batch;
            if comptime!(reduce_batch) { source_batch = sample; }
            let mut sr = F::new(0.0); let mut si = F::new(0.0);
            if alpha_r != F::new(0.0) || alpha_i != F::new(0.0) {
                for tile_k in 0..inner_tiles {
                    let ak = tile_k * 16usize + tx; let bk = tile_k * 16usize + ty;
                    let mut avr = F::new(0.0); let mut avi = F::new(0.0);
                    let mut bvr = F::new(0.0); let mut bvi = F::new(0.0);
                    if row < rows && ak < inner {
                        avr = ar[source_batch * ar.stride(0) + row * ar.stride(1) + ak * ar.stride(2)];
                        avi = ai[source_batch * ai.stride(0) + row * ai.stride(1) + ak * ai.stride(2)];
                    }
                    if bk < inner && col < columns {
                        bvr = br[source_batch * br.stride(0) + bk * br.stride(1) + col * br.stride(2)];
                        bvi = bi[source_batch * bi.stride(0) + bk * bi.stride(1) + col * bi.stride(2)];
                    }
                    tile_ar[tid] = avr; tile_ai[tid] = avi; tile_br[tid] = bvr; tile_bi[tid] = bvi;
                    sync_ruda();
                    for k in 0..16usize {
                        if tile_k * 16usize + k < inner {
                            let x = tile_ar[ty * 16usize + k]; let u = tile_ai[ty * 16usize + k];
                            let y = tile_br[k * 16usize + tx]; let v = tile_bi[k * 16usize + tx];
                            sr += x * y - u * v; si += x * v + u * y;
                        }
                    }
                    sync_ruda();
                }
            }
            real += alpha_r * sr - alpha_i * si; imag += alpha_r * si + alpha_i * sr;
        }
    }
    if valid { yr[address(yr, p)] = real; yi[address(yi, p)] = imag; }
}

pub(super) fn dispatch(xs: &[View], ints: &[i64], scalars: &[f64]) {
    assert_eq!(xs.len(), 4); assert!(ints.len() == 2 || ints.len() == 3); assert_eq!(scalars.len(), 4);
    let activation = *ints.get(2).unwrap_or(&0); assert!((0..=2).contains(&activation));
    let mode = ints[0]; assert!((0..=2).contains(&mode)); assert!(ints[1] == 0 || ints[1] == 1);
    let zero_beta = ints[1] != 0; let (bias, a, b, out) = (&xs[0], &xs[1], &xs[2], &xs[3]);
    let code = a.dtype; assert!(matches!(code, 0 | 1 | 2 | 9 | 10 | 11)); assert_eq!(code, b.dtype);
    assert_eq!(bias.dtype, out.dtype); assert_eq!(bias.shape, out.shape);
    assert!(out.dtype == code || mode == 1 && matches!(code, 1 | 2) && out.dtype == 0);
    assert!(activation == 0 || mode == 1 && matches!(code, 0 | 1 | 2 | 9) && out.dtype == code);
    let (row_tiles, column_tiles, inner_tiles, blocks) = if mode == 0 {
        assert_eq!(a.shape.len(), 2); assert_eq!(b.shape.len(), 1); assert_eq!(a.shape[1], b.shape[0]);
        assert_eq!(out.shape, vec![a.shape[0]]);
        (0, 0, 0, a.shape[0])
    } else {
        assert_eq!(a.shape.len(), 3); assert_eq!(b.shape.len(), 3);
        assert_eq!(a.shape[0], b.shape[0]); assert_eq!(a.shape[2], b.shape[1]);
        let expected = if mode == 1 { vec![a.shape[0], a.shape[1], b.shape[2]] } else { vec![a.shape[1], b.shape[2]] };
        assert_eq!(out.shape, expected);
        let rows = a.shape[1].div_ceil(16); let columns = b.shape[2].div_ceil(16);
        let batches = if mode == 1 { a.shape[0] } else { 1 };
        let blocks = rows.checked_mul(columns).and_then(|n| n.checked_mul(batches)).expect("BLAS accumulation grid overflow");
        (rows, columns, a.shape[2].div_ceil(16), blocks)
    };
    if out.len == 0 { return; }
    assert!(blocks <= u32::MAX as usize);
    let c = client(); let x = blocks.min(131072);
    let grid = RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1); let dim = RudaDim::new_1d(256);
    if matches!(code, 10 | 11) {
        let parts: Vec<View> = xs.iter().flat_map(|v| [scientific::component(v, false), scientific::component(v, true)]).collect();
        for part in &parts { scientific::indexable(part); }
        macro_rules! run { ($f:ty) => { unsafe {
            if mode == 0 {
                complex_gemv::launch::<$f, CudaRuntime>(&c, grid, dim,
                    parts[0].arg(), parts[1].arg(), parts[2].arg(), parts[3].arg(), parts[4].arg(), parts[5].arg(), parts[6].arg(), parts[7].arg(),
                    scalars[0] as $f, scalars[1] as $f, scalars[2] as $f, scalars[3] as $f, zero_beta);
            } else {
                complex_batched::launch::<$f, CudaRuntime>(&c, grid, dim,
                    parts[0].arg(), parts[1].arg(), parts[2].arg(), parts[3].arg(), parts[4].arg(), parts[5].arg(), parts[6].arg(), parts[7].arg(),
                    scalars[0] as $f, scalars[1] as $f, scalars[2] as $f, scalars[3] as $f,
                    row_tiles, column_tiles, inner_tiles, mode == 2, zero_beta);
            }
        }}; }
        match code { 10 => run!(f32), 11 => run!(f64), _ => unreachable!() }
    } else {
        assert_eq!(scalars[1], 0.0); assert_eq!(scalars[3], 0.0);
        macro_rules! vector { ($s:ty, $f:ty) => { unsafe {
            gemv::launch::<$s, $f, CudaRuntime>(&c, grid, dim, bias.arg(), a.arg(), b.arg(), out.arg(),
                scalars[0] as $f, scalars[2] as $f, zero_beta);
        }}; }
        macro_rules! matrix { ($s:ty, $o:ty, $f:ty) => { unsafe {
            batched::launch::<$s, $o, $f, CudaRuntime>(&c, grid, dim, bias.arg(), a.arg(), b.arg(), out.arg(),
                scalars[0] as $f, scalars[2] as $f, row_tiles, column_tiles, inner_tiles, mode == 2, zero_beta, activation as u32);
        }}; }
        if mode == 0 {
            match code { 0 => vector!(f32, f32), 1 => vector!(f16, f32), 2 => vector!(bf16, f32), 9 => vector!(f64, f64), _ => unreachable!() }
        } else {
            match (code, out.dtype) {
                (0, 0) => matrix!(f32, f32, f32), (1, 1) => matrix!(f16, f16, f32), (2, 2) => matrix!(bf16, bf16, f32),
                (1, 0) => matrix!(f16, f32, f32), (2, 0) => matrix!(bf16, f32, f32), (9, 9) => matrix!(f64, f64, f64),
                _ => unreachable!(),
            }
        }
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed); finish_dispatch(&c);
}
