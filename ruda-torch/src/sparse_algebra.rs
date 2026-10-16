use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn block_id() -> usize { RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize }

#[ruda]
fn equal(indices: &Tensor<i64>, a: usize, b: usize) -> bool {
    let mut same = true;
    for d in 0..indices.shape(0) {
        same = same && indices[d * indices.stride(0) + a * indices.stride(1)]
            == indices[d * indices.stride(0) + b * indices.stride(1)];
    }
    same
}

#[ruda(launch)]
fn dense_add<S: Numeric + RudaElement, A: Numeric + RudaElement>(
    indices: &Tensor<i64>, vr: &Tensor<S>, vi: &Tensor<S>, order: &Tensor<u32>,
    dr: &mut Tensor<S>, di: &mut Tensor<S>, ar: &Tensor<S>, ai: &Tensor<S>, width: usize,
    #[comptime] complex: bool, #[comptime] boolean: bool,
) {
    let p = block_id() * 256usize + UNIT_POS as usize;
    if p < vr.len() {
        let row = p / width; let column = p % width; let source = order[row] as usize;
        let mut first = row == 0usize;
        if row > 0usize { first = !equal(indices, order[row - 1usize] as usize, source); }
        if first {
            let mut ro = 0usize; let mut io = 0usize; let mut valid = true;
            for d in 0..indices.shape(0) {
                let coordinate = indices[d * indices.stride(0) + source * indices.stride(1)];
                if coordinate < 0i64 || coordinate >= i64::cast_from(dr.shape(d)) { valid = false; }
                ro += usize::cast_from(coordinate) * dr.stride(d); io += usize::cast_from(coordinate) * di.stride(d);
            }
            let mut rem = column; let mut d = dr.rank();
            while d > indices.shape(0) {
                d -= 1usize; let coordinate = rem % dr.shape(d); rem /= dr.shape(d);
                ro += coordinate * dr.stride(d); io += coordinate * di.stride(d);
            }
            if valid {
                let alpha_r = A::cast_from(ar[0]); let mut alpha_i = A::cast_from(0u32);
                if comptime!(complex) { alpha_i = A::cast_from(ai[0]); }
                let mut r = dr[ro]; let mut i = S::cast_from(0u32);
                if comptime!(complex) { i = di[io]; }
                let mut j = row; let mut more = true;
                while j < order.len() && more {
                    let entry = order[j] as usize;
                    if !equal(indices, source, entry) { more = false; }
                    else {
                        let x = A::cast_from(vr[address(vr, entry * width + column)]);
                        if comptime!(boolean) { r = S::cast_from(r != S::cast_from(0u32) || alpha_r != A::cast_from(0u32) && x != A::cast_from(0u32)); }
                        else if comptime!(complex) {
                            let y = A::cast_from(vi[address(vi, entry * width + column)]);
                            let pr = S::cast_from(alpha_r * x - alpha_i * y);
                            let pi = S::cast_from(alpha_r * y + alpha_i * x);
                            r = S::cast_from(A::cast_from(r) + A::cast_from(pr));
                            i = S::cast_from(A::cast_from(i) + A::cast_from(pi));
                        } else {
                            let product = S::cast_from(alpha_r * x);
                            r = S::cast_from(A::cast_from(r) + A::cast_from(product));
                        }
                    }
                    j += 1usize;
                }
                dr[ro] = r; if comptime!(complex) { di[io] = i; }
            }
        }
    }
}

#[ruda]
fn row_lower(indices: &Tensor<i64>, row: i64, #[comptime] upper: bool) -> usize {
    let mut lo = 0usize; let mut hi = indices.shape(1);
    while lo < hi {
        let mid = lo + (hi - lo) / 2usize; let key = indices[mid * indices.stride(1)];
        let mut before = key < row;
        if comptime!(upper) { before = key <= row; }
        if before { lo = mid + 1usize; } else { hi = mid; }
    }
    lo
}

#[ruda(launch)]
fn rows(indices: &Tensor<i64>, offsets: &mut Tensor<u32>) {
    let row = block_id() * 256usize + UNIT_POS as usize;
    if row < offsets.len() { offsets[row] = u32::cast_from(row_lower(indices, i64::cast_from(row), false)); }
}

#[ruda(launch)]
fn spmm<S: Float + RudaElement, F: Float + RudaElement>(
    indices: &Tensor<i64>, offsets: &Tensor<u32>, ar: &Tensor<S>, ai: &Tensor<S>,
    br: &Tensor<S>, bi: &Tensor<S>, cr: &Tensor<S>, ci: &Tensor<S>, yr: &mut Tensor<S>, yi: &mut Tensor<S>,
    alpha_r: F, alpha_i: F, beta_r: F, beta_i: F, tiles: usize, #[comptime] complex: bool,
) {
    let block = block_id(); let row = block / tiles; let tile = block % tiles;
    if row >= yr.shape(0) { return; }
    let tid = UNIT_POS as usize; let lane = tid % 32usize; let warp = tid / 32usize;
    let column = tile * 32usize + lane;
    let mut r = F::new(0.0); let mut i = F::new(0.0);
    if column < yr.shape(1) && (alpha_r != F::new(0.0) || alpha_i != F::new(0.0)) {
        let mut j = offsets[row] as usize + warp; let end = offsets[row + 1usize] as usize;
        while j < end {
            let k = indices[indices.stride(0) + j * indices.stride(1)];
            if k >= 0i64 && k < i64::cast_from(br.shape(0)) {
                let x = F::cast_from(ar[address(ar, j)]);
                let y = F::cast_from(br[k as usize * br.stride(0) + column * br.stride(1)]);
                if comptime!(complex) {
                    let u = F::cast_from(ai[address(ai, j)]);
                    let v = F::cast_from(bi[k as usize * bi.stride(0) + column * bi.stride(1)]);
                    r += x * y - u * v; i += x * v + u * y;
                } else { r = fma(x, y, r); }
            }
            j += 8usize;
        }
    }
    let mut partial_r = SharedMemory::<F>::new(256usize);
    let mut partial_i = SharedMemory::<F>::new(256usize);
    partial_r[tid] = r;
    if comptime!(complex) { partial_i[tid] = i; }
    sync_ruda();
    if tid < 32usize && column < yr.shape(1) {
        r = F::new(0.0); i = F::new(0.0);
        for w in 0..8usize {
            r += partial_r[w * 32usize + lane];
            if comptime!(complex) { i += partial_i[w * 32usize + lane]; }
        }
        let mut real = alpha_r * r; let mut imag = F::new(0.0);
        if comptime!(complex) { real -= alpha_i * i; imag = alpha_r * i + alpha_i * r; }
        if beta_r != F::new(0.0) || beta_i != F::new(0.0) {
            let x = F::cast_from(cr[row * cr.stride(0) + column * cr.stride(1)]);
            if comptime!(complex) {
                let y = F::cast_from(ci[row * ci.stride(0) + column * ci.stride(1)]);
                real += beta_r * x - beta_i * y; imag += beta_r * y + beta_i * x;
            } else { real += beta_r * x; }
        }
        yr[row * yr.stride(0) + column * yr.stride(1)] = S::cast_from(real);
        if comptime!(complex) { yi[row * yi.stride(0) + column * yi.stride(1)] = S::cast_from(imag); }
    }
}

#[ruda(launch)]
fn product_counts(a: &Tensor<i64>, b: &Tensor<i64>, begin: &mut Tensor<u32>, prefix: &mut Tensor<u64>) {
    let p = block_id() * 256usize + UNIT_POS as usize;
    if p < prefix.len() {
        let row = a[a.stride(0) + p * a.stride(1)];
        let start = row_lower(b, row, false); let end = row_lower(b, row, true);
        begin[p] = u32::cast_from(start); prefix[p] = u64::cast_from(end - start);
    }
}

#[ruda]
fn warp_prefix(value: u64) -> u64 {
    let mut result = value; let mut offset = 1u32;
    while offset < 32u32 {
        let previous = plane_shuffle_up(result, offset);
        if UNIT_POS % 32u32 >= offset { result += previous; }
        offset *= 2u32;
    }
    result
}

#[ruda(launch)]
fn scan(values: &mut Tensor<u64>, totals: &mut Tensor<u64>) {
    let block = block_id(); let tid = UNIT_POS as usize; let p = block * 256usize + tid;
    if block < totals.len() {
        let mut value = 0u64; if p < values.len() { value = values[p]; }
        value = warp_prefix(value);
        let mut warps = SharedMemory::<u64>::new(8usize);
        if tid % 32usize == 31usize { warps[tid / 32usize] = value; }
        sync_ruda();
        if tid < 32usize {
            let mut subtotal = 0u64; if tid < 8usize { subtotal = warps[tid]; }
            subtotal = warp_prefix(subtotal); if tid < 8usize { warps[tid] = subtotal; }
        }
        sync_ruda();
        if tid >= 32usize { value += warps[tid / 32usize - 1usize]; }
        if p < values.len() { values[p] = value; }
        if tid == 255usize { totals[block] = value; }
    }
}

#[ruda(launch)]
fn add_totals(values: &mut Tensor<u64>, totals: &Tensor<u64>) {
    let p = block_id() * 256usize + UNIT_POS as usize;
    if p < values.len() && p >= 256usize { values[p] += totals[p / 256usize - 1usize]; }
}

#[ruda(launch)]
fn product_total(prefix: &Tensor<u64>, total: &mut Tensor<u64>) {
    if UNIT_POS == 0u32 { total[0] = prefix[prefix.len() - 1usize]; }
}

#[ruda(launch)]
fn products<S: Float + RudaElement, F: Float + RudaElement>(
    a: &Tensor<i64>, ar: &Tensor<S>, ai: &Tensor<S>, b: &Tensor<i64>, br: &Tensor<S>, bi: &Tensor<S>,
    begin: &Tensor<u32>, prefix: &Tensor<u64>, indices: &mut Tensor<i64>, vr: &mut Tensor<F>, vi: &mut Tensor<F>,
    #[comptime] complex: bool,
) {
    let p = block_id() * 256usize + UNIT_POS as usize;
    if p < vr.len() {
        let mut lo = 0usize; let mut hi = prefix.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2usize;
            if prefix[mid] <= u64::cast_from(p) { lo = mid + 1usize; } else { hi = mid; }
        }
        let mut previous = 0u64; if lo > 0usize { previous = prefix[lo - 1usize]; }
        let j = begin[lo] as usize + usize::cast_from(u64::cast_from(p) - previous);
        indices[p * indices.stride(1)] = a[lo * a.stride(1)];
        indices[indices.stride(0) + p * indices.stride(1)] = b[b.stride(0) + j * b.stride(1)];
        let x = F::cast_from(ar[address(ar, lo)]); let y = F::cast_from(br[address(br, j)]);
        let mut r = x * y; let mut i = F::new(0.0);
        if comptime!(complex) {
            let u = F::cast_from(ai[address(ai, lo)]); let v = F::cast_from(bi[address(bi, j)]);
            r -= u * v; i = x * v + u * y;
        }
        vr[address(vr, p)] = r; if comptime!(complex) { vi[address(vi, p)] = i; }
    }
}

fn grid(blocks: usize) -> RudaCount {
    assert!(blocks > 0 && blocks <= u32::MAX as usize);
    let x = blocks.min(131072); RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1)
}

fn packed(v: &View, n: usize, dtype: u32) {
    assert_eq!(v.dtype, dtype); assert_eq!(v.shape, vec![n]); assert_eq!(v.strides, vec![1]);
}

fn part(v: &View, imaginary: bool) -> View {
    if matches!(v.dtype,10 | 11 | 14) { scientific::component(v, imaginary) }
    else { View { handle: v.handle.clone(), shape: v.shape.clone(), strides: v.strides.clone(), len: v.len, dtype: v.dtype } }
}

fn matrix(indices: &View, values: &View) {
    assert_eq!(indices.dtype,4); assert_eq!(indices.shape.len(),2); assert_eq!(indices.shape[0],2);
    assert_eq!(values.shape, vec![indices.shape[1]]); assert!(matches!(values.dtype,0 | 1 | 2 | 9 | 10 | 11));
}

pub(super) fn dispatch(op: u32, xs: &[View], scalars: &[f64]) {
    let c = client(); let dim = RudaDim::new_1d(256);
    match op {
        508 => {
            assert_eq!(xs.len(),5); assert!(scalars.is_empty());
            let (indices, values, order, dense, alpha) = (&xs[0],&xs[1],&xs[2],&xs[3],&xs[4]);
            assert_eq!(indices.dtype,4); assert_eq!(indices.shape.len(),2);
            let sd = indices.shape[0]; let n = indices.shape[1];
            assert!(sd <= dense.shape.len()); assert_eq!(values.shape[0],n);
            assert_eq!(values.shape[1..],dense.shape[sd..]); packed(order,n,5);
            assert_eq!(values.dtype,dense.dtype); assert_eq!(values.dtype,alpha.dtype); assert_eq!(alpha.len,1);
            if values.len == 0 { return; }
            let width: usize = values.shape[1..].iter().product();
            let parts: Vec<View> = [values,dense,alpha].iter().flat_map(|v| [part(v,false),part(v,true)]).collect();
            for v in &parts { scientific::indexable(v); }
            macro_rules! run { ($s:ty,$a:ty) => { unsafe { dense_add::launch::<$s,$a,CudaRuntime>(&c,
                grid(values.len.div_ceil(256)),dim,indices.arg(),parts[0].arg(),parts[1].arg(),order.arg(),
                parts[2].arg(),parts[3].arg(),parts[4].arg(),parts[5].arg(),width,
                matches!(values.dtype,10 | 11 | 14),values.dtype==3); }}; }
            match values.dtype { 0 | 10 => run!(f32,f32),1 | 14 => run!(f16,f32),2 => run!(bf16,f32),
                9 | 11 => run!(f64,f64),3 | 8 => run!(u8,u64),4 => run!(i64,u64),5 => run!(i32,u64),
                6 => run!(i16,u64),7 => run!(i8,u64),_ => panic!("unsupported sparse add dtype") }
            LAUNCHES.fetch_add(1,Ordering::Relaxed);
        }
        509 => {
            assert_eq!(xs.len(),5); assert_eq!(scalars.len(),4); matrix(&xs[0],&xs[1]);
            let (b,bias,out) = (&xs[2],&xs[3],&xs[4]);
            assert_eq!(b.shape.len(),2); assert_eq!(out.shape.len(),2); assert_eq!(b.shape[1],out.shape[1]);
            assert_eq!(bias.shape,out.shape);
            for v in [b,bias,out] { assert_eq!(v.dtype,xs[1].dtype); }
            if out.len==0 { return; }
            let count=out.shape[0].checked_add(1).expect("sparse row count overflow");
            let offsets=View::packed(c.empty(count.checked_mul(4).expect("sparse row workspace overflow")),vec![count],5);
            scientific::indexable(&offsets);
            unsafe { rows::launch::<CudaRuntime>(&c,grid(count.div_ceil(256)),dim,xs[0].arg(),offsets.arg()); }
            let parts: Vec<View> = [&xs[1],b,bias,out].iter().flat_map(|v| [part(v,false),part(v,true)]).collect();
            for v in &parts { scientific::indexable(v); }
            let tiles=out.shape[1].div_ceil(32);
            let blocks=out.shape[0].checked_mul(tiles).expect("sparse matmul launch overflow");
            macro_rules! run { ($s:ty,$f:ty) => { unsafe { spmm::launch::<$s,$f,CudaRuntime>(&c,grid(blocks),dim,
                xs[0].arg(),offsets.arg(),parts[0].arg(),parts[1].arg(),parts[2].arg(),parts[3].arg(),parts[4].arg(),parts[5].arg(),
                parts[6].arg(),parts[7].arg(),scalars[0] as $f,scalars[1] as $f,scalars[2] as $f,scalars[3] as $f,tiles,
                matches!(xs[1].dtype,10 | 11)); }}; }
            match xs[1].dtype {0 | 10 => run!(f32,f32),1 => run!(f16,f32),2 => run!(bf16,f32),9 | 11 => run!(f64,f64),_ => unreachable!()}
            LAUNCHES.fetch_add(2,Ordering::Relaxed);
        }
        510 => {
            assert_eq!(xs.len(),5); assert!(scalars.is_empty());
            for v in &xs[..2] { assert_eq!(v.dtype,4); assert_eq!(v.shape.len(),2); assert_eq!(v.shape[0],2); }
            let n=xs[0].shape[1]; assert!(n>0); packed(&xs[2],n,5); packed(&xs[3],n,4); packed(&xs[4],1,4);
            unsafe { product_counts::launch::<CudaRuntime>(&c,grid(n.div_ceil(256)),dim,xs[0].arg(),xs[1].arg(),xs[2].arg(),xs[3].arg()); }
            let mut levels=Vec::new(); let mut items=n; let mut launches=1u64;
            loop {
                let blocks=items.div_ceil(256);
                levels.push(View::packed(c.empty(blocks.checked_mul(8).expect("sparse prefix workspace overflow")),vec![blocks],4));
                let level=levels.len()-1; let values=if level==0 {&xs[3]} else {&levels[level-1]};
                unsafe { scan::launch::<CudaRuntime>(&c,grid(blocks),dim,values.arg(),levels[level].arg()); }
                launches+=1; if blocks==1 {break;} items=blocks;
            }
            for level in (0..levels.len()-1).rev() {
                let values=if level==0 {&xs[3]} else {&levels[level-1]};
                unsafe { add_totals::launch::<CudaRuntime>(&c,grid(values.len.div_ceil(256)),dim,values.arg(),levels[level].arg()); }
                launches+=1;
            }
            unsafe { product_total::launch::<CudaRuntime>(&c,RudaCount::Static(1,1,1),dim,xs[3].arg(),xs[4].arg()); }
            LAUNCHES.fetch_add(launches+1,Ordering::Relaxed);
        }
        511 => {
            assert_eq!(xs.len(),8); assert!(scalars.is_empty()); matrix(&xs[0],&xs[1]); matrix(&xs[2],&xs[3]);
            assert_eq!(xs[1].dtype,xs[3].dtype); let n=xs[0].shape[1]; packed(&xs[4],n,5); packed(&xs[5],n,4);
            let size=xs[7].len; assert_eq!(xs[6].dtype,4); assert_eq!(xs[6].shape,vec![2,size]); assert_eq!(xs[7].shape,vec![size]);
            assert_eq!(xs[7].dtype,if matches!(xs[1].dtype,1 | 2) {0} else {xs[1].dtype});
            if size==0 {return;}
            let parts: Vec<View> = [&xs[1],&xs[3],&xs[7]].iter().flat_map(|v| [part(v,false),part(v,true)]).collect();
            for v in &parts { scientific::indexable(v); }
            macro_rules! run { ($s:ty,$f:ty) => { unsafe { products::launch::<$s,$f,CudaRuntime>(&c,grid(size.div_ceil(256)),dim,
                xs[0].arg(),parts[0].arg(),parts[1].arg(),xs[2].arg(),parts[2].arg(),parts[3].arg(),
                xs[4].arg(),xs[5].arg(),xs[6].arg(),parts[4].arg(),parts[5].arg(),matches!(xs[1].dtype,10 | 11)); }}; }
            match xs[1].dtype {0 | 10 => run!(f32,f32),1 => run!(f16,f32),2 => run!(bf16,f32),9 | 11 => run!(f64,f64),_ => unreachable!()}
            LAUNCHES.fetch_add(1,Ordering::Relaxed);
        }
        _ => unreachable!(),
    }
    finish_dispatch(&c);
}
