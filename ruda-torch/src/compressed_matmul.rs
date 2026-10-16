use super::*;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn multiply<I: Numeric + RudaElement, S: Float + RudaElement, F: Float + RudaElement>(
    pointers: &Tensor<I>, indices: &Tensor<I>, ar: &Tensor<S>, ai: &Tensor<S>,
    br: &Tensor<S>, bi: &Tensor<S>, cr: &Tensor<S>, ci: &Tensor<S>, yr: &mut Tensor<S>, yi: &mut Tensor<S>,
    alpha_r: F, alpha_i: F, beta_r: F, beta_i: F, tiles: usize,
    #[comptime] block_rows: usize, #[comptime] block_cols: usize,
    #[comptime] complex: bool, #[comptime] vector: bool,
) {
    let block = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    let batch = block / (yr.shape(1) * tiles);
    if batch >= yr.shape(0) { terminate!(); }
    let row = block / tiles % yr.shape(1);
    let tid = UNIT_POS as usize; let lane = tid % 32usize;
    let mut column = block % tiles * 32usize + lane;
    let mut entry_lane = tid / 32usize; let mut step = 8usize;
    if comptime!(vector) { column = 0usize; entry_lane = tid; step = 256usize; }
    let mut r = F::new(0.0); let mut i = F::new(0.0);
    if column < yr.shape(2) && (alpha_r != F::new(0.0) || alpha_i != F::new(0.0)) {
        let compressed_row = row / block_rows; let inner_row = row % block_rows;
        let pointer = batch * pointers.stride(0) + compressed_row * pointers.stride(1);
        let begin = usize::cast_from(pointers[pointer]) * block_cols;
        let end = usize::cast_from(pointers[pointer + pointers.stride(1)]) * block_cols;
        let count = end - begin; let mut offset = entry_lane; let mut active = offset < count;
        while active {
            let term = begin + offset; let entry = term / block_cols; let inner_col = term % block_cols;
            let plain = i64::cast_from(indices[batch * indices.stride(0) + entry * indices.stride(1)]);
            let k = plain * i64::cast_from(block_cols) + i64::cast_from(inner_col);
            if k >= 0i64 && k < i64::cast_from(br.shape(1)) {
                let ap = batch * ar.stride(0) + entry * ar.stride(1) + inner_row * ar.stride(2) + inner_col * ar.stride(3);
                let bp = batch * br.stride(0) + k as usize * br.stride(1) + column * br.stride(2);
                let x = F::cast_from(ar[ap]); let y = F::cast_from(br[bp]);
                if comptime!(complex) {
                    let aip = batch * ai.stride(0) + entry * ai.stride(1) + inner_row * ai.stride(2) + inner_col * ai.stride(3);
                    let bip = batch * bi.stride(0) + k as usize * bi.stride(1) + column * bi.stride(2);
                    let u = F::cast_from(ai[aip]); let v = F::cast_from(bi[bip]);
                    r += x * y - u * v; i += x * v + u * y;
                } else { r = fma(x,y,r); }
            }
            if count - offset <= step { active = false; } else { offset += step; }
        }
    }
    let mut write = tid < 32usize && column < yr.shape(2);
    if comptime!(vector) {
        r = crate::blas_accumulate::block_sum::<F>(r);
        if comptime!(complex) { i = crate::blas_accumulate::block_sum::<F>(i); }
        write = tid == 0usize;
    } else {
        let mut partial_r = SharedMemory::<F>::new(256usize);
        let mut partial_i = SharedMemory::<F>::new(256usize);
        partial_r[tid] = r;
        if comptime!(complex) { partial_i[tid] = i; }
        sync_ruda();
        r = F::new(0.0); i = F::new(0.0);
        if write {
            for warp in 0..8usize {
                r += partial_r[warp * 32usize + lane];
                if comptime!(complex) { i += partial_i[warp * 32usize + lane]; }
            }
        }
    }
    if write {
        let mut real = alpha_r * r; let mut imag = F::new(0.0);
        if comptime!(complex) { real -= alpha_i * i; imag = alpha_r * i + alpha_i * r; }
        if beta_r != F::new(0.0) || beta_i != F::new(0.0) {
            let cp = batch * cr.stride(0) + row * cr.stride(1) + column * cr.stride(2);
            let x = F::cast_from(cr[cp]);
            if comptime!(complex) {
                let cip = batch * ci.stride(0) + row * ci.stride(1) + column * ci.stride(2);
                let y = F::cast_from(ci[cip]);
                real += beta_r * x - beta_i * y; imag += beta_r * y + beta_i * x;
            } else { real += beta_r * x; }
        }
        yr[batch * yr.stride(0) + row * yr.stride(1) + column * yr.stride(2)] = S::cast_from(real);
        if comptime!(complex) {
            yi[batch * yi.stride(0) + row * yi.stride(1) + column * yi.stride(2)] = S::cast_from(imag);
        }
    }
}

fn part(v: &View, imaginary: bool) -> View {
    if matches!(v.dtype,10 | 11) { scientific::component(v,imaginary) }
    else { View {handle:v.handle.clone(),shape:v.shape.clone(),strides:v.strides.clone(),len:v.len,dtype:v.dtype} }
}

pub(super) fn dispatch(xs: &[View], scalars: &[f64]) {
    assert_eq!(xs.len(),6); assert_eq!(scalars.len(),4);
    let (pointers,indices,values,dense,bias,out) = (&xs[0],&xs[1],&xs[2],&xs[3],&xs[4],&xs[5]);
    assert!(pointers.dtype == 4 || pointers.dtype == 5); assert_eq!(pointers.dtype,indices.dtype);
    assert_eq!(pointers.shape.len(),2); assert_eq!(indices.shape.len(),2); assert_eq!(values.shape.len(),4);
    for v in [dense,bias,out] {assert_eq!(v.shape.len(),3);assert_eq!(v.dtype,values.dtype);}
    assert!(matches!(values.dtype,0 | 1 | 2 | 9 | 10 | 11)); assert_eq!(bias.shape,out.shape);
    let batch = out.shape[0]; let rows = out.shape[1]; let columns = out.shape[2];
    let block_rows = values.shape[2]; let block_cols = values.shape[3];
    assert!(block_rows > 0 && block_cols > 0); assert_eq!(rows % block_rows,0); assert_eq!(dense.shape[1] % block_cols,0);
    assert_eq!(pointers.shape,[batch,rows/block_rows+1]); assert_eq!(indices.shape,[batch,values.shape[1]]);
    assert_eq!(values.shape[0],batch); assert_eq!(dense.shape[0],batch); assert_eq!(dense.shape[2],columns);
    if out.len == 0 {return;}
    let tiles = columns.div_ceil(32);
    let blocks = batch.checked_mul(rows).and_then(|n|n.checked_mul(tiles)).expect("compressed matmul grid overflow");
    assert!(blocks > 0 && blocks <= u32::MAX as usize);
    let x = blocks.min(131072); let grid = RudaCount::Static(x as u32,blocks.div_ceil(x) as u32,1);
    let c = client(); let dim = RudaDim::new_1d(256);
    let parts: Vec<View> = [values,dense,bias,out].iter().flat_map(|v|[part(v,false),part(v,true)]).collect();
    for v in &parts {scientific::indexable(v);}
    macro_rules! launch {
        ($i:ty,$s:ty,$f:ty) => {unsafe {
            multiply::launch::<$i,$s,$f,CudaRuntime>(&c,grid.clone(),dim,pointers.arg(),indices.arg(),
                parts[0].arg(),parts[1].arg(),parts[2].arg(),parts[3].arg(),parts[4].arg(),parts[5].arg(),parts[6].arg(),parts[7].arg(),
                scalars[0] as $f,scalars[1] as $f,scalars[2] as $f,scalars[3] as $f,tiles,
                block_rows,block_cols,matches!(values.dtype,10 | 11),columns == 1);
        }};
    }
    macro_rules! index {($s:ty,$f:ty) => {if pointers.dtype == 4 {launch!(i64,$s,$f);} else {launch!(i32,$s,$f);}};}
    match values.dtype {0 | 10=>index!(f32,f32),1=>index!(f16,f32),2=>index!(bf16,f32),9 | 11=>index!(f64,f64),_=>unreachable!()}
    LAUNCHES.fetch_add(1,Ordering::Relaxed); finish_dispatch(&c);
}
