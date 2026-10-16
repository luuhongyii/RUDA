use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn position() -> usize {
    (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize
}

#[ruda(launch)]
fn block_keys<I: Numeric + RudaElement>(pointers: &Tensor<I>, plain: &Tensor<I>, keys: &mut Tensor<i64>,
    compressed_block: i64, plain_block: i64, #[comptime] column: bool,
) {
    let p = position();
    if p < plain.len() {
        let mut lo = 0usize; let mut hi = pointers.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2usize;
            if i64::cast_from(pointers[address(pointers, mid)]) <= i64::cast_from(p) { lo = mid + 1usize; }
            else { hi = mid; }
        }
        let c = i64::cast_from(lo - 1usize); let v = i64::cast_from(plain[address(plain, p)]);
        let mut inner = (c % compressed_block) * plain_block + v % plain_block;
        if comptime!(column) { inner = (v % plain_block) * compressed_block + c % compressed_block; }
        keys[p * keys.stride(0)] = c / compressed_block;
        keys[p * keys.stride(0) + keys.stride(1)] = v / plain_block;
        keys[p * keys.stride(0) + 2usize * keys.stride(1)] = inner;
    }
}

#[ruda(launch)]
fn mark_blocks(keys: &Tensor<i64>, order: &Tensor<u32>, groups: &mut Tensor<u32>) {
    let p = position();
    if p < order.len() {
        let mut first = p == 0usize;
        if p > 0usize {
            let a = order[p - 1usize] as usize * keys.stride(0);
            let b = order[p] as usize * keys.stride(0);
            first = keys[a] != keys[b] || keys[a + keys.stride(1)] != keys[b + keys.stride(1)];
        }
        groups[p] = u32::cast_from(first);
    }
}

#[ruda(launch)]
fn block_starts(groups: &Tensor<u32>, starts: &mut Tensor<u32>, count: &mut Tensor<i64>) {
    let p = position();
    if p < groups.len() {
        let mut first = p == 0usize;
        if p > 0usize { first = groups[p] != groups[p - 1usize]; }
        if first { starts[groups[p] as usize - 1usize] = u32::cast_from(p); }
        if p + 1usize == groups.len() { count[0] = i64::cast_from(groups[p]); }
    }
}

#[ruda(launch)]
fn block_indices(keys: &Tensor<i64>, order: &Tensor<u32>, starts: &Tensor<u32>, output: &mut Tensor<i64>) {
    let p = position();
    if p < output.len() {
        let axis = p / output.shape(1); let block = p % output.shape(1);
        let source = order[starts[block] as usize] as usize;
        output[address(output, p)] = keys[source * keys.stride(0) + axis * keys.stride(1)];
    }
}

#[ruda(launch)]
fn match_entries(keys: &Tensor<i64>, order: &Tensor<u32>, starts: &Tensor<u32>, matched: &mut Tensor<u32>,
    block_count: usize, area: usize,
) {
    let p = position();
    if p < matched.len() {
        let block = p / area; let inner = i64::cast_from(p % area);
        let start = starts[block] as usize;
        let mut end = order.len();
        if block + 1usize < block_count { end = starts[block + 1usize] as usize; }
        let mut lo = start; let mut hi = end;
        while lo < hi {
            let mid = lo + (hi - lo) / 2usize;
            let from = order[mid] as usize * keys.stride(0) + 2usize * keys.stride(1);
            if keys[from] <= inner { lo = mid + 1usize; } else { hi = mid; }
        }
        let mut found = 4294967295u32;
        if lo > start {
            let source = order[lo - 1usize];
            if keys[source as usize * keys.stride(0) + 2usize * keys.stride(1)] == inner { found = source; }
        }
        matched[p] = found;
    }
}

#[ruda(launch)]
fn copy_values(source: &Tensor<u8>, matched: &Tensor<u32>, output: &mut Tensor<u8>, width: usize) {
    let p = position();
    if p < output.len() {
        let from = matched[p / width]; let mut value = 0u8;
        if from != 4294967295u32 { value = source[address(source, from as usize * width + p % width)]; }
        output[address(output, p)] = value;
    }
}

fn grid(work: usize) -> RudaCount {
    let blocks = work.div_ceil(256); assert!(blocks > 0 && blocks <= u32::MAX as usize);
    let x = blocks.min(131072); RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1)
}

fn vector(view: &View, length: usize, dtype: u32) {
    assert_eq!(view.dtype,dtype); assert_eq!(view.shape,[length]); assert_eq!(view.strides,[1]);
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64]) {
    let c = client(); let block = RudaDim::new_1d(256);
    if op == 519 {
        assert_eq!(xs.len(),7); assert_eq!(ints.len(),3);
        assert!(ints[0] > 0 && ints[1] > 0 && (ints[2] == 0 || ints[2] == 1));
        assert!(ints[0].checked_mul(ints[1]).is_some());
        let (pointers,plain,keys,order,groups,starts,count) = (&xs[0],&xs[1],&xs[2],&xs[3],&xs[4],&xs[5],&xs[6]);
        assert!(pointers.dtype == 4 || pointers.dtype == 5); assert_eq!(pointers.dtype,plain.dtype);
        assert_eq!(pointers.shape.len(),1); assert_eq!(plain.shape.len(),1);
        let n = plain.len; assert!(n > 0 && pointers.len > 1);
        assert_eq!(keys.dtype,4); assert_eq!(keys.shape,[n,3]); assert_eq!(keys.strides,[3,1]);
        for v in [order,groups,starts] { vector(v,n,5); } vector(count,1,4);
        unsafe {
            if pointers.dtype == 4 {
                block_keys::launch::<i64,CudaRuntime>(&c,grid(n),block,pointers.arg(),plain.arg(),keys.arg(),ints[0],ints[1],ints[2] != 0);
            } else {
                block_keys::launch::<i32,CudaRuntime>(&c,grid(n),block,pointers.arg(),plain.arg(),keys.arg(),ints[0],ints[1],ints[2] != 0);
            }
        }
        let (_sort_workspace,sorts) = crate::unique_ops::stable_order(&c,keys,order,3,false);
        unsafe { mark_blocks::launch::<CudaRuntime>(&c,grid(n),block,keys.arg(),order.arg(),groups.arg()); }
        let (_levels,scans) = crate::unique_ops::inclusive_prefix(&c,groups);
        unsafe { block_starts::launch::<CudaRuntime>(&c,grid(n),block,groups.arg(),starts.arg(),count.arg()); }
        LAUNCHES.fetch_add(3+sorts+scans,Ordering::Relaxed); finish_dispatch(&c);
    } else {
        assert_eq!(op,520); assert_eq!(xs.len(),6); assert!(ints.is_empty());
        let (keys,order,starts,values,indices,out) = (&xs[0],&xs[1],&xs[2],&xs[3],&xs[4],&xs[5]);
        assert_eq!(keys.dtype,4); assert_eq!(keys.shape.len(),2); assert_eq!(keys.shape[1],3);
        let n = keys.shape[0]; vector(order,n,5); vector(starts,n,5);
        assert_eq!(indices.dtype,4); assert_eq!(indices.shape.len(),2); assert_eq!(indices.shape[0],2);
        let blocks = indices.shape[1]; assert!(blocks <= n);
        assert!(!values.shape.is_empty()); assert_eq!(values.shape[0],n);
        assert!(out.shape.len() >= 3); assert_eq!(out.shape[0],blocks);
        assert!(out.shape[1] > 0 && out.shape[2] > 0); assert_eq!(out.shape[3..],values.shape[1..]);
        assert_eq!(out.dtype,values.dtype); assert!(matches!(out.dtype,0..=11 | 14));
        if blocks == 0 { return; }
        unsafe { block_indices::launch::<CudaRuntime>(&c,grid(indices.len),block,keys.arg(),order.arg(),starts.arg(),indices.arg()); }
        let mut launches = 1u64;
        if out.len > 0 {
            let area = out.shape[1].checked_mul(out.shape[2]).expect("sparse block area overflow");
            let cells = blocks.checked_mul(area).expect("sparse block map overflow");
            let source_bytes = values.byte_view(); let output_bytes = out.byte_view();
            scientific::indexable(&source_bytes); scientific::indexable(&output_bytes);
            let matched = View::packed(c.empty(cells.checked_mul(4).expect("sparse block map byte overflow")),vec![cells],5);
            scientific::indexable(&matched);
            let width = source_bytes.len / n; assert!(width > 0);
            unsafe {
                match_entries::launch::<CudaRuntime>(&c,grid(cells),block,keys.arg(),order.arg(),starts.arg(),matched.arg(),blocks,area);
                copy_values::launch::<CudaRuntime>(&c,grid(output_bytes.len),block,source_bytes.arg(),matched.arg(),output_bytes.arg(),width);
            }
            launches += 2;
        }
        LAUNCHES.fetch_add(launches,Ordering::Relaxed); finish_dispatch(&c);
    }
}
