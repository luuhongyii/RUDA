use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn position() -> usize {
    (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize
}

#[ruda(launch)]
fn gather(indices: &Tensor<i64>, dense: &Tensor<u8>, output_indices: &mut Tensor<i64>,
    filtered: &mut Tensor<u8>, width: usize, leading: usize,
) {
    let p = position(); let nnz = indices.shape(1); let sd = indices.shape(0);
    if p < output_indices.len() {
        let d = p / output_indices.shape(1); let row = p % output_indices.shape(1);
        let source = row % nnz; let mut coordinate = 0i64;
        if d < leading {
            let mut batch = row / nnz; let mut axis = leading;
            while axis > d { axis -= 1usize; coordinate = i64::cast_from(batch % dense.shape(axis)); batch /= dense.shape(axis); }
        } else { coordinate = indices[(d - leading) * indices.stride(0) + source * indices.stride(1)]; }
        output_indices[address(output_indices, p)] = coordinate;
    }
    if p < filtered.len() {
        let row = p / width; let source = row % nnz;
        let mut offset = 0usize; let mut batch = row / nnz; let mut axis = leading;
        while axis > 0usize { axis -= 1usize; offset += (batch % dense.shape(axis)) * dense.stride(axis); batch /= dense.shape(axis); }
        let mut valid = true;
        for d in 0..sd {
            let coordinate = indices[d * indices.stride(0) + source * indices.stride(1)];
            if coordinate < 0i64 || coordinate >= i64::cast_from(dense.shape(leading + d)) { valid = false; }
            offset += usize::cast_from(coordinate) * dense.stride(leading + d);
        }
        let mut rem = p % width; axis = dense.rank();
        while axis > leading + sd {
            axis -= 1usize; offset += (rem % dense.shape(axis)) * dense.stride(axis); rem /= dense.shape(axis);
        }
        let mut value = 0u8; if valid { value = dense[offset]; }
        filtered[address(filtered, p)] = value;
    }
}

#[ruda]
fn compare(a: &Tensor<i64>, ai: usize, b: &Tensor<i64>, bi: usize) -> i32 {
    let mut d = 0usize; let mut result = 0i32;
    while d < a.shape(0) && result == 0i32 {
        let x = a[d * a.stride(0) + ai * a.stride(1)];
        let y = b[d * b.stride(0) + bi * b.stride(1)];
        if x < y { result = -1i32; } else if x > y { result = 1i32; }
        d += 1usize;
    }
    result
}

#[ruda(launch)]
fn bounds(search: &Tensor<i64>, source: &Tensor<i64>, order: &Tensor<u32>,
    begin: &mut Tensor<u32>, end: &mut Tensor<u32>, #[comptime] identity: bool,
) {
    let p = position();
    if p < begin.len() {
        let mut lo = 0usize; let mut hi = search.shape(1);
        while lo < hi {
            let mid = lo + (hi - lo) / 2usize; let mut j = mid;
            if comptime!(!identity) { j = order[mid] as usize; }
            if compare(search, j, source, p) < 0i32 { lo = mid + 1usize; } else { hi = mid; }
        }
        begin[p] = u32::cast_from(lo); hi = search.shape(1);
        while lo < hi {
            let mid = lo + (hi - lo) / 2usize; let mut j = mid;
            if comptime!(!identity) { j = order[mid] as usize; }
            if compare(search, j, source, p) <= 0i32 { lo = mid + 1usize; } else { hi = mid; }
        }
        end[p] = u32::cast_from(lo);
    }
}

#[ruda(launch)]
fn intersection<S: Numeric + RudaElement, A: Numeric + RudaElement>(
    sr: &Tensor<S>, si: &Tensor<S>, tr: &Tensor<S>, ti: &Tensor<S>, order: &Tensor<u32>,
    begin: &Tensor<u32>, end: &Tensor<u32>, yr: &mut Tensor<S>, yi: &mut Tensor<S>, width: usize,
    #[comptime] identity: bool, #[comptime] complex: bool, #[comptime] boolean: bool,
) {
    let p = position();
    if p < yr.len() {
        let row = p / width; let column = p % width;
        let a = A::cast_from(sr[address(sr, p)]); let mut b = A::cast_from(0u32);
        if comptime!(complex) { b = A::cast_from(si[address(si, p)]); }
        let mut r = A::cast_from(0u32); let mut i = A::cast_from(0u32);
        let mut j = begin[row] as usize;
        while j < end[row] as usize {
            let mut entry = j; if comptime!(!identity) { entry = order[j] as usize; }
            let q = entry * width + column;
            let c = A::cast_from(tr[address(tr, q)]);
            if comptime!(boolean) { r = A::cast_from(r != A::cast_from(0u32) || a != A::cast_from(0u32) && c != A::cast_from(0u32)); }
            else if comptime!(complex) {
                let d = A::cast_from(ti[address(ti, q)]);
                r += a * c - b * d; i += a * d + b * c;
            } else { r += a * c; }
            j += 1usize;
        }
        yr[address(yr, p)] = S::cast_from(r);
        if comptime!(complex) { yi[address(yi, p)] = S::cast_from(i); }
    }
}

fn grid(work: usize) -> RudaCount {
    let blocks = work.div_ceil(256); assert!(blocks > 0 && blocks <= u32::MAX as usize);
    let x = blocks.min(131072); RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1)
}

fn packed(value: &View, n: usize) {
    assert_eq!(value.dtype,5); assert_eq!(value.shape,vec![n]); assert_eq!(value.strides,vec![1]);
}

fn indices(value: &View) {
    assert_eq!(value.dtype,4); assert_eq!(value.shape.len(),2);
}

fn part(value: &View, imaginary: bool) -> View {
    if matches!(value.dtype,10 | 11 | 14) { scientific::component(value,imaginary) }
    else { View {handle:value.handle.clone(),shape:value.shape.clone(),strides:value.strides.clone(),len:value.len,dtype:value.dtype} }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64]) {
    let c = client(); let block = RudaDim::new_1d(256);
    match op {
        512 => {
            assert_eq!(xs.len(),4); assert_eq!(ints.len(),1); assert!(ints[0]>=0);
            let leading=ints[0] as usize; let (source,dense,target,filtered)=(&xs[0],&xs[1],&xs[2],&xs[3]);
            indices(source); indices(target); let sd=source.shape[0]; let nnz=source.shape[1];
            assert!(nnz>0); assert_eq!(target.shape[0],sd+leading); assert!(dense.shape.len()>=sd+leading);
            assert_eq!(filtered.dtype,dense.dtype); assert_eq!(filtered.shape[0],target.shape[1]);
            assert_eq!(filtered.shape[1..],dense.shape[sd+leading..]);
            let batches=dense.shape[..leading].iter().try_fold(1usize,|n,&s|n.checked_mul(s)).expect("sparse product batch overflow");
            assert_eq!(target.shape[1],nnz.checked_mul(batches).expect("sparse product nnz overflow"));
            let input_bytes=dense.byte_view(); let output_bytes=filtered.byte_view();
            scientific::indexable(&input_bytes); scientific::indexable(&output_bytes);
            let work=target.len.max(output_bytes.len); if work==0 {return;}
            let width=output_bytes.shape[1..].iter().try_fold(1usize,|n,&s|n.checked_mul(s)).expect("sparse product block overflow");
            unsafe {gather::launch::<CudaRuntime>(&c,grid(work),block,source.arg(),input_bytes.arg(),target.arg(),output_bytes.arg(),width,leading);}
            LAUNCHES.fetch_add(1,Ordering::Relaxed);
        }
        513 => {
            assert_eq!(xs.len(),5); assert_eq!(ints.len(),1); assert!(matches!(ints[0],0 | 1));
            indices(&xs[0]); indices(&xs[1]); assert_eq!(xs[0].shape[0],xs[1].shape[0]);
            let n=xs[1].shape[1]; let identity=ints[0]!=0;
            packed(&xs[2],if identity {0} else {xs[0].shape[1]}); packed(&xs[3],n); packed(&xs[4],n);
            if n==0 {return;}
            unsafe {bounds::launch::<CudaRuntime>(&c,grid(n),block,xs[0].arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),xs[4].arg(),identity);}
            LAUNCHES.fetch_add(1,Ordering::Relaxed);
        }
        514 => {
            assert_eq!(xs.len(),6); assert_eq!(ints.len(),1); assert!(matches!(ints[0],0 | 1));
            let (source,search,out)=(&xs[0],&xs[1],&xs[5]); let identity=ints[0]!=0;
            assert!(!source.shape.is_empty()); assert!(!search.shape.is_empty());
            assert_eq!(source.shape,out.shape); assert_eq!(source.shape[1..],search.shape[1..]);
            assert_eq!(source.dtype,search.dtype); assert_eq!(source.dtype,out.dtype);
            packed(&xs[2],if identity {0} else {search.shape[0]}); packed(&xs[3],source.shape[0]); packed(&xs[4],source.shape[0]);
            if out.len==0 {return;}
            let width=out.shape[1..].iter().try_fold(1usize,|n,&s|n.checked_mul(s)).expect("sparse product block overflow");
            let parts:Vec<View>=[source,search,out].iter().flat_map(|v|[part(v,false),part(v,true)]).collect();
            for v in &parts {scientific::indexable(v);}
            macro_rules! run {($s:ty,$a:ty)=>{unsafe {intersection::launch::<$s,$a,CudaRuntime>(&c,grid(out.len),block,
                parts[0].arg(),parts[1].arg(),parts[2].arg(),parts[3].arg(),xs[2].arg(),xs[3].arg(),xs[4].arg(),
                parts[4].arg(),parts[5].arg(),width,identity,matches!(out.dtype,10 | 11 | 14),out.dtype==3);}};}
            match out.dtype {0 | 10=>run!(f32,f32),1 | 14=>run!(f16,f32),2=>run!(bf16,f32),9 | 11=>run!(f64,f64),
                3 | 8=>run!(u8,u64),4=>run!(i64,u64),5=>run!(i32,u64),6=>run!(i16,u64),7=>run!(i8,u64),_=>panic!("unsupported sparse product dtype")}
            LAUNCHES.fetch_add(1,Ordering::Relaxed);
        }
        _=>unreachable!(),
    }
    finish_dispatch(&c);
}
