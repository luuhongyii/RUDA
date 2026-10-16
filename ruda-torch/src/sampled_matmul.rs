use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn sampled<I: Numeric + RudaElement, F: Float + RudaElement>(
    pointers: &Tensor<I>, indices: &Tensor<I>, ar: &Tensor<F>, ai: &Tensor<F>, br: &Tensor<F>, bi: &Tensor<F>,
    cr: &Tensor<F>, ci: &Tensor<F>, yr: &mut Tensor<F>, yi: &mut Tensor<F>,
    alpha_r: F, alpha_i: F, beta_r: F, beta_i: F, #[comptime] complex: bool,
) {
    let block = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    let lane = UNIT_POS as usize % 32usize;
    let p = block * 8usize + UNIT_POS as usize / 32usize;
    if p >= yr.len() { terminate!(); }
    let batch_rank = yr.rank() - 1usize; let nnz = yr.shape(batch_rank);
    let batch = p / nnz; let entry = p % nnz;
    let rows = ar.shape(batch_rank); let inner = ar.shape(batch_rank + 1usize);
    let mut a_base = 0usize; let mut b_base = 0usize; let mut remaining = batch; let mut axis = batch_rank;
    while axis > 0usize {
        axis -= 1usize;
        let coordinate = remaining % yr.shape(axis); remaining /= yr.shape(axis);
        a_base += coordinate * ar.stride(axis); b_base += coordinate * br.stride(axis);
    }
    let pointer_base = batch * (rows + 1usize);
    let mut lo = 0usize; let mut hi = rows + 1usize;
    while lo < hi {
        let mid = lo + (hi - lo) / 2usize;
        if i64::cast_from(pointers[address(pointers, pointer_base + mid)]) <= i64::cast_from(entry) { lo = mid + 1usize; }
        else { hi = mid; }
    }
    let row = lo - 1usize;
    let column = usize::cast_from(indices[address(indices, p)]);
    let mut real = F::new(0.0); let mut imag = F::new(0.0);
    if alpha_r != F::new(0.0) || alpha_i != F::new(0.0) {
        let mut k = lane; let mut active = k < inner;
        while active {
            let ap = a_base + row * ar.stride(batch_rank) + k * ar.stride(batch_rank + 1usize);
            let bp = b_base + k * br.stride(batch_rank) + column * br.stride(batch_rank + 1usize);
            let x = ar[ap]; let y = br[bp];
            if comptime!(complex) {
                let u = ai[ap]; let v = bi[bp];
                real += x * y - u * v; imag += x * v + u * y;
            } else { real = fma(x,y,real); }
            if inner - k <= 32usize { active = false; } else { k += 32usize; }
        }
    }
    let mut offset = 16u32;
    while offset > 0u32 {
        real += plane_shuffle_down(real,offset);
        if comptime!(complex) { imag += plane_shuffle_down(imag,offset); }
        offset /= 2u32;
    }
    if lane == 0usize {
        let mut r = alpha_r * real; let mut i = F::new(0.0);
        if comptime!(complex) { r -= alpha_i * imag; i = alpha_r * imag + alpha_i * real; }
        if beta_r != F::new(0.0) || beta_i != F::new(0.0) {
            let x = cr[address(cr,p)];
            if comptime!(complex) {
                let y = ci[address(ci,p)];
                r += beta_r * x - beta_i * y; i += beta_r * y + beta_i * x;
            } else { r += beta_r * x; }
        }
        yr[address(yr,p)] = r;
        if comptime!(complex) { yi[address(yi,p)] = i; }
    }
}

fn part(v: &View, imaginary: bool) -> View {
    if matches!(v.dtype,10 | 11) { scientific::component(v,imaginary) }
    else {View {handle:v.handle.clone(),shape:v.shape.clone(),strides:v.strides.clone(),len:v.len,dtype:v.dtype}}
}

pub(super) fn dispatch(xs: &[View], scalars: &[f64]) {
    assert_eq!(xs.len(),6); assert_eq!(scalars.len(),4);
    let (pointers,indices,a,b,bias,out) = (&xs[0],&xs[1],&xs[2],&xs[3],&xs[4],&xs[5]);
    assert!(pointers.dtype == 4 || pointers.dtype == 5); assert_eq!(indices.dtype,pointers.dtype);
    assert!(!out.shape.is_empty()); let rank = out.shape.len()-1;
    assert_eq!(pointers.shape.len(),rank+1); assert_eq!(indices.shape,out.shape); assert_eq!(bias.shape,out.shape);
    assert_eq!(a.shape.len(),rank+2); assert_eq!(b.shape.len(),rank+2);
    assert_eq!(pointers.shape[..rank],out.shape[..rank]);
    assert_eq!(a.shape[..rank],out.shape[..rank]); assert_eq!(b.shape[..rank],out.shape[..rank]);
    assert_eq!(pointers.shape[rank],a.shape[rank]+1); assert_eq!(a.shape[rank+1],b.shape[rank]);
    assert!(matches!(out.dtype,0 | 9 | 10 | 11));
    for v in [a,b,bias] {assert_eq!(v.dtype,out.dtype);}
    if out.len == 0 {return;}
    assert!(a.shape[rank] > 0 && a.shape[rank+1] > 0 && b.shape[rank+1] > 0);
    let blocks = out.len.div_ceil(8); let x = blocks.min(131072);
    let grid = RudaCount::Static(x as u32,blocks.div_ceil(x) as u32,1);
    let c = client(); let dim = RudaDim::new_1d(256);
    let parts: Vec<View> = [a,b,bias,out].iter().flat_map(|v|[part(v,false),part(v,true)]).collect();
    for v in &parts {scientific::indexable(v);}
    macro_rules! launch {($i:ty,$f:ty) => {unsafe {
        sampled::launch::<$i,$f,CudaRuntime>(&c,grid.clone(),dim,pointers.arg(),indices.arg(),
            parts[0].arg(),parts[1].arg(),parts[2].arg(),parts[3].arg(),parts[4].arg(),parts[5].arg(),parts[6].arg(),parts[7].arg(),
            scalars[0] as $f,scalars[1] as $f,scalars[2] as $f,scalars[3] as $f,matches!(out.dtype,10 | 11));
    }};}
    macro_rules! index {($f:ty) => {if pointers.dtype == 4 {launch!(i64,$f);} else {launch!(i32,$f);}};}
    if matches!(out.dtype,0 | 10) {index!(f32);} else {index!(f64);}
    LAUNCHES.fetch_add(1,Ordering::Relaxed); finish_dispatch(&c);
}
