use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn thread_id() -> usize {
    (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize
}

#[ruda(launch)]
fn count_union(ap: &Tensor<i64>, aj: &Tensor<i64>, bp: &Tensor<i64>, bj: &Tensor<i64>, counts: &mut Tensor<i64>) {
    let p = thread_id(); let rows = ap.shape(1)-1usize;
    if p < ap.shape(0)*rows {
        let batch = p/rows; let row = p%rows; let base = batch*(rows+1usize);
        let mut a = usize::cast_from(ap[address(ap,base+row)]); let ae = usize::cast_from(ap[address(ap,base+row+1usize)]);
        let mut b = usize::cast_from(bp[address(bp,base+row)]); let be = usize::cast_from(bp[address(bp,base+row+1usize)]);
        let mut count = 0i64;
        while a < ae || b < be {
            let mut x = 9223372036854775807i64; let mut y = x;
            if a < ae {x = aj[address(aj,batch*aj.shape(1)+a)];}
            if b < be {y = bj[address(bj,batch*bj.shape(1)+b)];}
            if x <= y {a += 1usize;}
            if y <= x {b += 1usize;}
            count += 1i64;
        }
        if row == 0usize {counts[base] = 0i64;}
        counts[base+row+1usize] = count;
    }
}

#[ruda(launch)]
fn emit_union<S: Numeric + RudaElement, A: Numeric + RudaElement>(
    ap: &Tensor<i64>, aj: &Tensor<i64>, bp: &Tensor<i64>, bj: &Tensor<i64>, pointers: &Tensor<i64>,
    ar: &Tensor<S>, ai: &Tensor<S>, br: &Tensor<S>, bi: &Tensor<S>,
    indices: &mut Tensor<i64>, vr: &mut Tensor<S>, vi: &mut Tensor<S>, alpha_r: &Tensor<S>, alpha_i: &Tensor<S>,
    span: usize, #[comptime] complex: bool, #[comptime] boolean: bool,
) {
    let p = thread_id(); let rows = ap.shape(1)-1usize; let width = vr.shape(2);
    if p < ap.shape(0)*rows*span {
        let batch = p/span/rows; let row = p/span%rows; let payload = p%span; let base = batch*(rows+1usize);
        let mut a = usize::cast_from(ap[address(ap,base+row)]); let ae = usize::cast_from(ap[address(ap,base+row+1usize)]);
        let mut b = usize::cast_from(bp[address(bp,base+row)]); let be = usize::cast_from(bp[address(bp,base+row+1usize)]);
        let mut out = usize::cast_from(pointers[base+row]);
        let alpha = A::cast_from(alpha_r[0]); let mut imaginary = A::cast_from(0u32);
        if comptime!(complex) {imaginary = A::cast_from(alpha_i[0]);}
        while a < ae || b < be {
            let mut x = 9223372036854775807i64; let mut y = x;
            if a < ae {x = aj[address(aj,batch*aj.shape(1)+a)];}
            if b < be {y = bj[address(bj,batch*bj.shape(1)+b)];}
            let has_a = x <= y; let has_b = y <= x;
            let mut key = x; if y < x {key = y;}
            if payload == 0usize {indices[batch*indices.shape(1)+out] = key;}
            if payload < width {
                let mut real = A::cast_from(0u32); let mut imag = A::cast_from(0u32);
                if has_a {
                    let pos = (batch*aj.shape(1)+a)*width+payload;
                    real = A::cast_from(ar[address(ar,pos)]);
                    if comptime!(complex) {imag = A::cast_from(ai[address(ai,pos)]);}
                }
                if has_b {
                    let pos = (batch*bj.shape(1)+b)*width+payload;
                    let value = A::cast_from(br[address(br,pos)]);
                    if comptime!(boolean) {real = A::cast_from(real != A::cast_from(0u32) || alpha != A::cast_from(0u32) && value != A::cast_from(0u32));}
                    else if comptime!(complex) {
                        let value_i = A::cast_from(bi[address(bi,pos)]);
                        real += alpha*value-imaginary*value_i; imag += alpha*value_i+imaginary*value;
                    } else {real += alpha*value;}
                }
                let pos = (batch*indices.shape(1)+out)*width+payload;
                vr[address(vr,pos)] = S::cast_from(real);
                if comptime!(complex) {vi[address(vi,pos)] = S::cast_from(imag);}
            }
            if has_a {a += 1usize;} if has_b {b += 1usize;} out += 1usize;
        }
    }
}

#[ruda(launch)]
fn dense_add<S: Numeric + RudaElement, A: Numeric + RudaElement>(
    pointers: &Tensor<i64>, indices: &Tensor<i64>, vr: &Tensor<S>, vi: &Tensor<S>,
    out_r: &mut Tensor<S>, out_i: &mut Tensor<S>, alpha_r: &Tensor<S>, alpha_i: &Tensor<S>,
    #[comptime] column: bool, #[comptime] complex: bool, #[comptime] boolean: bool,
) {
    let p = thread_id(); let rows = pointers.shape(1)-1usize; let width = vr.shape(4);
    let mut compressed_block = vr.shape(2); let mut plain_block = vr.shape(3);
    if comptime!(column) {compressed_block = vr.shape(3); plain_block = vr.shape(2);}
    if p < pointers.shape(0)*rows*compressed_block*width {
        let payload = p%width; let local = p/width%compressed_block;
        let row = p/width/compressed_block%rows; let batch = p/width/compressed_block/rows;
        let base = batch*(rows+1usize);
        let begin = usize::cast_from(pointers[address(pointers,base+row)]);
        let end = usize::cast_from(pointers[address(pointers,base+row+1usize)]);
        let alpha = A::cast_from(alpha_r[0]); let mut imaginary = A::cast_from(0u32);
        if comptime!(complex) {imaginary = A::cast_from(alpha_i[0]);}
        for entry in begin..end {
            let plain = usize::cast_from(indices[address(indices,batch*indices.shape(1)+entry)]);
            for cell in 0..plain_block {
                let mut r = row*compressed_block+local; let mut c = plain*plain_block+cell;
                let mut br = local; let mut bc = cell;
                if comptime!(column) {r = plain*plain_block+cell; c = row*compressed_block+local; br = cell; bc = local;}
                let value_pos = ((((batch*indices.shape(1)+entry)*vr.shape(2)+br)*vr.shape(3)+bc)*width)+payload;
                let out_pos = (((batch*out_r.shape(1)+r)*out_r.shape(2)+c)*width)+payload;
                let ro = address(out_r,out_pos); let x = A::cast_from(vr[address(vr,value_pos)]);
                let value = A::cast_from(out_r[ro]);
                if comptime!(boolean) {out_r[ro] = S::cast_from(value != A::cast_from(0u32) || alpha != A::cast_from(0u32) && x != A::cast_from(0u32));}
                else if comptime!(complex) {
                    let io = address(out_i,out_pos); let y = A::cast_from(vi[address(vi,value_pos)]);
                    let pr = S::cast_from(alpha*x-imaginary*y); let pi = S::cast_from(alpha*y+imaginary*x);
                    out_r[ro] = S::cast_from(value+A::cast_from(pr));
                    out_i[io] = S::cast_from(A::cast_from(out_i[io])+A::cast_from(pi));
                } else {
                    let product = S::cast_from(alpha*x);
                    out_r[ro] = S::cast_from(value+A::cast_from(product));
                }
            }
        }
    }
}

fn part(v: &View, imaginary: bool) -> View {
    if matches!(v.dtype,10 | 11 | 14) {scientific::component(v,imaginary)}
    else {View {handle:v.handle.clone(),shape:v.shape.clone(),strides:v.strides.clone(),len:v.len,dtype:v.dtype}}
}

fn grid(items: usize) -> RudaCount {
    assert!(items > 0 && items <= u32::MAX as usize);
    let blocks = items.div_ceil(256); let x = blocks.min(131072);
    RudaCount::Static(x as u32,blocks.div_ceil(x) as u32,1)
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64]) {
    let c = client(); let dim = RudaDim::new_1d(256);
    if op == 525 || op == 526 {
        assert!(ints.is_empty()); assert_eq!(xs.len(),if op == 525 {5} else {10});
        for v in &xs[..5] {assert_eq!(v.dtype,4); assert_eq!(v.shape.len(),2);}
        assert_eq!(xs[0].shape,xs[2].shape); assert_eq!(xs[0].shape,xs[4].shape);
        let batches = xs[0].shape[0]; assert!(xs[0].shape[1] > 0); let rows = xs[0].shape[1]-1;
        assert_eq!(xs[1].shape[0],batches); assert_eq!(xs[3].shape[0],batches);
        assert_eq!(xs[4].strides,[rows+1,1]);
        let work = batches.checked_mul(rows).expect("compressed union work overflow");
        if op == 525 {
            if work == 0 {return;}
            unsafe {count_union::launch::<CudaRuntime>(&c,grid(work),dim,xs[0].arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),xs[4].arg());}
        } else {
            let dtype = xs[8].dtype; assert!(matches!(dtype,0..=11 | 14));
            assert_eq!(xs[8].shape.len(),3); let width = xs[8].shape[2]; let nnz = xs[8].shape[1];
            assert_eq!(xs[8].shape[0],batches);
            for (v,n) in [(&xs[5],xs[1].shape[1]),(&xs[6],xs[3].shape[1])] {assert_eq!(v.dtype,dtype); assert_eq!(v.shape,[batches,n,width]);}
            assert_eq!(xs[7].dtype,4); assert_eq!(xs[7].shape,[batches,nnz]); assert_eq!(xs[7].strides,[nnz,1]);
            assert_eq!(xs[9].dtype,dtype); assert_eq!(xs[9].len,1);
            let span = width.max(1); let work = work.checked_mul(span).expect("compressed union payload work overflow");
            if work == 0 || nnz == 0 {return;}
            let parts: Vec<View> = [&xs[5],&xs[6],&xs[8],&xs[9]].iter().flat_map(|v|[part(v,false),part(v,true)]).collect();
            for v in &parts {scientific::indexable(v);}
            macro_rules! run {($s:ty,$a:ty) => {unsafe {emit_union::launch::<$s,$a,CudaRuntime>(&c,grid(work),dim,
                xs[0].arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),xs[4].arg(),parts[0].arg(),parts[1].arg(),parts[2].arg(),parts[3].arg(),
                xs[7].arg(),parts[4].arg(),parts[5].arg(),parts[6].arg(),parts[7].arg(),span,matches!(dtype,10 | 11 | 14),dtype==3);}};}
            match dtype {0 | 10 => run!(f32,f32),1 | 14 => run!(f16,f32),2 => run!(bf16,f32),9 | 11 => run!(f64,f64),
                3 | 8 => run!(u8,u64),4 => run!(i64,u64),5 => run!(i32,u64),6 => run!(i16,u64),7 => run!(i8,u64),_ => unreachable!()}
        }
    } else {
        assert_eq!(op,527); assert_eq!(xs.len(),5); assert_eq!(ints.len(),1); assert!(ints[0] == 0 || ints[0] == 1);
        let (pointers,indices,values,out,alpha) = (&xs[0],&xs[1],&xs[2],&xs[3],&xs[4]);
        for v in [pointers,indices] {assert_eq!(v.dtype,4); assert_eq!(v.shape.len(),2);}
        assert!(pointers.shape[1] > 0); let batches = pointers.shape[0]; let rows = pointers.shape[1]-1;
        assert_eq!(indices.shape[0],batches); assert_eq!(values.shape.len(),5); assert_eq!(out.shape.len(),4);
        assert_eq!(values.shape[..2],indices.shape); assert_eq!(out.shape[0],batches); assert_eq!(values.shape[4],out.shape[3]);
        let column = ints[0] != 0; let cb = values.shape[if column {3} else {2}]; let pb = values.shape[if column {2} else {3}];
        assert!(cb > 0 && pb > 0); assert_eq!(out.shape[if column {2} else {1}],rows*cb);
        assert_eq!(out.shape[if column {1} else {2}]%pb,0);
        let dtype = out.dtype; assert!(matches!(dtype,0..=11 | 14)); assert_eq!(values.dtype,dtype); assert_eq!(alpha.dtype,dtype); assert_eq!(alpha.len,1);
        let work = batches.checked_mul(rows).and_then(|n|n.checked_mul(cb)).and_then(|n|n.checked_mul(values.shape[4])).expect("compressed dense-add work overflow");
        if work == 0 || values.len == 0 {return;}
        let parts: Vec<View> = [values,out,alpha].iter().flat_map(|v|[part(v,false),part(v,true)]).collect();
        for v in &parts {scientific::indexable(v);}
        macro_rules! run {($s:ty,$a:ty) => {unsafe {dense_add::launch::<$s,$a,CudaRuntime>(&c,grid(work),dim,
            pointers.arg(),indices.arg(),parts[0].arg(),parts[1].arg(),parts[2].arg(),parts[3].arg(),parts[4].arg(),parts[5].arg(),
            column,matches!(dtype,10 | 11 | 14),dtype==3);}};}
        match dtype {0 | 10 => run!(f32,f32),1 | 14 => run!(f16,f32),2 => run!(bf16,f32),9 | 11 => run!(f64,f64),
            3 | 8 => run!(u8,u64),4 => run!(i64,u64),5 => run!(i32,u64),6 => run!(i16,u64),7 => run!(i8,u64),_ => unreachable!()}
    }
    LAUNCHES.fetch_add(1,Ordering::Relaxed); finish_dispatch(&c);
}
