use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn row_id() -> usize {
    (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize
}

#[ruda]
fn column(indices: &Tensor<i64>, cursors: &Tensor<u32>, entry: usize) -> i64 {
    indices[address(indices, cursors[entry] as usize)]
}

#[ruda]
fn before(indices: &Tensor<i64>, cursors: &Tensor<u32>, a: usize, b: usize) -> bool {
    let x = column(indices,cursors,a); let y = column(indices,cursors,b);
    x < y || x == y && a < b
}

#[ruda]
fn sift(indices: &Tensor<i64>, cursors: &Tensor<u32>, heap: &mut Tensor<u32>, base: usize, root: usize, size: usize) {
    let entry = heap[base + root] as usize;
    let mut position = root; let mut child = position * 2usize + 1usize; let mut active = child < size;
    while active {
        if child + 1usize < size {
            if before(indices,cursors,heap[base + child + 1usize] as usize,heap[base + child] as usize) { child += 1usize; }
        }
        let next = heap[base + child] as usize;
        if before(indices,cursors,next,entry) {
            heap[base + position] = u32::cast_from(next); position = child; child = position * 2usize + 1usize;
            active = child < size;
        } else { active = false; }
    }
    heap[base + position] = u32::cast_from(entry);
}

#[ruda]
fn build(
    aj: &Tensor<i64>, bp: &Tensor<i64>, bj: &Tensor<i64>, heap: &mut Tensor<u32>,
    cursors: &mut Tensor<u32>, ends: &mut Tensor<u32>, start: usize, end: usize,
) -> usize {
    let mut size = 0usize;
    for entry in start..end {
        let k = usize::cast_from(aj[address(aj,entry)]);
        let begin = usize::cast_from(bp[address(bp,k)]); let stop = usize::cast_from(bp[address(bp,k + 1usize)]);
        cursors[entry] = u32::cast_from(begin); ends[entry] = u32::cast_from(stop);
        if begin < stop { heap[start + size] = u32::cast_from(entry); size += 1usize; }
    }
    let mut parent = size / 2usize;
    while parent > 0usize { parent -= 1usize; sift(bj,cursors,heap,start,parent,size); }
    size
}

#[ruda]
fn advance(bj: &Tensor<i64>, heap: &mut Tensor<u32>, cursors: &mut Tensor<u32>, ends: &Tensor<u32>, base: usize, size: usize) -> usize {
    let entry = heap[base] as usize; cursors[entry] += 1u32;
    let mut remaining = size;
    if cursors[entry] == ends[entry] {
        remaining -= 1usize;
        if remaining > 0usize { heap[base] = heap[base + remaining]; }
    }
    if remaining > 0usize { sift(bj,cursors,heap,base,0usize,remaining); }
    remaining
}

#[ruda]
fn next_column(bj: &Tensor<i64>, cj: &Tensor<i64>, heap: &Tensor<u32>, cursors: &Tensor<u32>, base: usize, size: usize, bias: usize, bias_end: usize) -> i64 {
    let mut next = 9223372036854775807i64;
    if size > 0usize { next = column(bj,cursors,heap[base] as usize); }
    if bias < bias_end {
        let candidate = cj[address(cj,bias)]; if candidate < next { next = candidate; }
    }
    next
}

#[ruda(launch)]
fn symbolic(
    ap: &Tensor<i64>, aj: &Tensor<i64>, bp: &Tensor<i64>, bj: &Tensor<i64>, cp: &Tensor<i64>, cj: &Tensor<i64>,
    heap: &mut Tensor<u32>, cursors: &mut Tensor<u32>, ends: &mut Tensor<u32>, counts: &mut Tensor<i64>,
) {
    let row = row_id();
    if row + 1usize < ap.len() {
        if row == 0usize { counts[0] = 0i64; }
        let start = usize::cast_from(ap[address(ap,row)]); let end = usize::cast_from(ap[address(ap,row + 1usize)]);
        let mut size = build(aj,bp,bj,heap,cursors,ends,start,end);
        let mut bias = usize::cast_from(cp[address(cp,row)]); let bias_end = usize::cast_from(cp[address(cp,row + 1usize)]);
        let mut count = 0i64;
        while size > 0usize || bias < bias_end {
            let next = next_column(bj,cj,heap,cursors,start,size,bias,bias_end);
            let mut more = size > 0usize;
            while more {
                if column(bj,cursors,heap[start] as usize) == next {
                    size = advance(bj,heap,cursors,ends,start,size); more = size > 0usize;
                } else { more = false; }
            }
            more = bias < bias_end;
            while more {
                if cj[address(cj,bias)] == next { bias += 1usize; more = bias < bias_end; }
                else { more = false; }
            }
            count += 1i64;
        }
        counts[row + 1usize] = count;
    }
}

#[ruda(launch)]
fn numeric<S: Float + RudaElement, F: Float + RudaElement>(
    ap: &Tensor<i64>, aj: &Tensor<i64>, bp: &Tensor<i64>, bj: &Tensor<i64>, cp: &Tensor<i64>, cj: &Tensor<i64>,
    heap: &mut Tensor<u32>, cursors: &mut Tensor<u32>, ends: &mut Tensor<u32>, pointers: &Tensor<i64>,
    ar: &Tensor<S>, ai: &Tensor<S>, br: &Tensor<S>, bi: &Tensor<S>, cr: &Tensor<S>, ci: &Tensor<S>,
    indices: &mut Tensor<i32>, vr: &mut Tensor<S>, vi: &mut Tensor<S>,
    alpha_r: F, alpha_i: F, beta_r: F, beta_i: F, #[comptime] complex: bool,
) {
    let row = row_id();
    if row + 1usize < ap.len() {
        let start = usize::cast_from(ap[address(ap,row)]); let end = usize::cast_from(ap[address(ap,row + 1usize)]);
        let mut size = build(aj,bp,bj,heap,cursors,ends,start,end);
        let mut bias = usize::cast_from(cp[address(cp,row)]); let bias_end = usize::cast_from(cp[address(cp,row + 1usize)]);
        let mut output = usize::cast_from(pointers[row]);
        while size > 0usize || bias < bias_end {
            let next = next_column(bj,cj,heap,cursors,start,size,bias,bias_end);
            let mut r = F::new(0.0); let mut i = F::new(0.0); let mut product = false;
            let mut more = size > 0usize;
            while more {
                let a = heap[start] as usize;
                if column(bj,cursors,a) == next {
                    product = true;
                    if alpha_r != F::new(0.0) || alpha_i != F::new(0.0) {
                        let b = cursors[a] as usize;
                        let x = F::cast_from(ar[address(ar,a)]); let y = F::cast_from(br[address(br,b)]);
                        if comptime!(complex) {
                            let u = F::cast_from(ai[address(ai,a)]); let v = F::cast_from(bi[address(bi,b)]);
                            r += x * y - u * v; i += x * v + u * y;
                        } else { r = fma(x,y,r); }
                    }
                    size = advance(bj,heap,cursors,ends,start,size); more = size > 0usize;
                } else { more = false; }
            }
            let mut real = F::new(0.0); let mut imag = F::new(0.0);
            if product {
                real = alpha_r * r;
                if comptime!(complex) { real -= alpha_i * i; imag = alpha_r * i + alpha_i * r; }
            }
            let mut bias_r = F::new(0.0); let mut bias_i = F::new(0.0); let mut has_bias = false;
            more = bias < bias_end;
            while more {
                if cj[address(cj,bias)] == next {
                    has_bias = true;
                    if beta_r != F::new(0.0) || beta_i != F::new(0.0) {
                        bias_r += F::cast_from(cr[address(cr,bias)]);
                        if comptime!(complex) { bias_i += F::cast_from(ci[address(ci,bias)]); }
                    }
                    bias += 1usize; more = bias < bias_end;
                } else { more = false; }
            }
            if has_bias {
                real += beta_r * bias_r;
                if comptime!(complex) { real -= beta_i * bias_i; imag += beta_r * bias_i + beta_i * bias_r; }
            }
            indices[output] = i32::cast_from(next); vr[address(vr,output)] = S::cast_from(real);
            if comptime!(complex) { vi[address(vi,output)] = S::cast_from(imag); }
            output += 1usize;
        }
    }
}

fn vector(v: &View, dtype: u32) { assert_eq!(v.dtype,dtype); assert_eq!(v.shape.len(),1); }

fn part(v: &View, imaginary: bool) -> View {
    if matches!(v.dtype,10 | 11) { scientific::component(v,imaginary) }
    else { View {handle:v.handle.clone(),shape:v.shape.clone(),strides:v.strides.clone(),len:v.len,dtype:v.dtype} }
}

pub(super) fn dispatch(op: u32, xs: &[View], scalars: &[f64]) {
    assert_eq!(xs.len(),if op == 523 {10} else {15}); assert_eq!(scalars.len(),if op == 523 {0} else {4});
    for v in &xs[..6] { vector(v,4); }
    assert!(xs[0].len > 0 && xs[2].len > 0); assert_eq!(xs[4].len,xs[0].len);
    let rows = xs[0].len-1;
    for v in &xs[6..9] {vector(v,5); assert_eq!(v.len,xs[1].len); assert_eq!(v.strides,[1]);}
    vector(&xs[9],4); assert_eq!(xs[9].len,rows+1); assert_eq!(xs[9].strides,[1]);
    assert!(rows <= i32::MAX as usize && xs[1].len <= i32::MAX as usize && xs[3].len <= i32::MAX as usize);
    if rows == 0 {return;}
    let c = client(); let dim = RudaDim::new_1d(256); let blocks = rows.div_ceil(256); let x = blocks.min(131072);
    let grid = RudaCount::Static(x as u32,blocks.div_ceil(x) as u32,1);
    if op == 523 {
        unsafe { symbolic::launch::<CudaRuntime>(&c,grid,dim,xs[0].arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),
            xs[4].arg(),xs[5].arg(),xs[6].arg(),xs[7].arg(),xs[8].arg(),xs[9].arg()); }
    } else {
        assert_eq!(op,524); let dtype = xs[14].dtype;
        assert!(matches!(dtype,0 | 1 | 2 | 9 | 10 | 11));
        for (v,n) in [(&xs[10],xs[1].len),(&xs[11],xs[3].len),(&xs[12],xs[5].len)] {vector(v,dtype); assert_eq!(v.len,n);}
        vector(&xs[13],5); vector(&xs[14],dtype); assert_eq!(xs[13].len,xs[14].len); assert_eq!(xs[13].strides,[1]);
        assert!(xs[14].len <= i32::MAX as usize);
        if xs[14].len == 0 {return;}
        let parts: Vec<View> = [&xs[10],&xs[11],&xs[12],&xs[14]].iter().flat_map(|v|[part(v,false),part(v,true)]).collect();
        for v in &parts {scientific::indexable(v);}
        macro_rules! run {($s:ty,$f:ty) => {unsafe {
            numeric::launch::<$s,$f,CudaRuntime>(&c,grid,dim,xs[0].arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),
                xs[4].arg(),xs[5].arg(),xs[6].arg(),xs[7].arg(),xs[8].arg(),xs[9].arg(),parts[0].arg(),parts[1].arg(),
                parts[2].arg(),parts[3].arg(),parts[4].arg(),parts[5].arg(),xs[13].arg(),parts[6].arg(),parts[7].arg(),
                scalars[0] as $f,scalars[1] as $f,scalars[2] as $f,scalars[3] as $f,matches!(dtype,10 | 11));
        }};}
        match dtype {0 | 10 => run!(f32,f32),1 => run!(f16,f32),2 => run!(bf16,f32),9 | 11 => run!(f64,f64),_ => unreachable!()}
    }
    LAUNCHES.fetch_add(1,Ordering::Relaxed); finish_dispatch(&c);
}
