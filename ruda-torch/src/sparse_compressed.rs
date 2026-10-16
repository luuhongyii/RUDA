use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn position() -> usize {
    (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize
}

#[ruda(launch)]
fn coo_to_csr<I: Numeric + RudaElement, O: Numeric + RudaElement>(
    rows: &Tensor<I>, pointers: &mut Tensor<O>,
) {
    let row = position();
    if row < pointers.len() {
        let mut lo = 0usize; let mut hi = rows.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2usize;
            if i64::cast_from(rows[address(rows, mid)]) < i64::cast_from(row) { lo = mid + 1usize; }
            else { hi = mid; }
        }
        pointers[address(pointers, row)] = O::cast_from(lo);
    }
}

#[ruda(launch)]
fn csr_to_coo<I: Numeric + RudaElement, O: Numeric + RudaElement>(
    pointers: &Tensor<I>, output: &mut Tensor<O>, nnz: usize, #[comptime] transpose: bool,
) {
    let p = position();
    if p < output.shape(1) {
        let batch = p / nnz; let entry = p % nnz;
        let batch_dim = pointers.rank() - 1usize;
        let width = pointers.shape(batch_dim);
        let base = batch * width;
        let mut lo = 0usize; let mut hi = width;
        while lo < hi {
            let mid = lo + (hi - lo) / 2usize;
            if i64::cast_from(pointers[address(pointers, base + mid)]) <= i64::cast_from(entry) { lo = mid + 1usize; }
            else { hi = mid; }
        }
        let mut row_axis = batch_dim;
        if comptime!(transpose) { row_axis += 1usize; }
        output[row_axis * output.stride(0) + p * output.stride(1)] = O::cast_from(lo - 1usize);
        let mut remaining = batch; let mut axis = batch_dim;
        while axis > 0usize {
            axis -= 1usize;
            output[axis * output.stride(0) + p * output.stride(1)] = O::cast_from(remaining % pointers.shape(axis));
            remaining /= pointers.shape(axis);
        }
    }
}

#[ruda(launch)]
fn validate<I: Numeric + RudaElement>(
    pointers: &Tensor<I>, indices: &Tensor<I>, status: &mut Tensor<Atomic<u32>>,
    cdim: usize, dim: i64, nnz: usize,
) {
    let p = position();
    if p < indices.len() {
        let index = i64::cast_from(indices[address(indices, p)]);
        if index < 0i64 || index >= dim { status[0].store(1u32); }
    }
    if p < pointers.len() {
        let axis = p % (cdim + 1usize); let batch = p / (cdim + 1usize);
        let start = i64::cast_from(pointers[address(pointers, p)]);
        if axis == 0usize && start != 0i64 { status[0].store(1u32); }
        if axis == cdim && start != i64::cast_from(nnz) { status[0].store(1u32); }
        if axis < cdim {
            let end = i64::cast_from(pointers[address(pointers, p + 1usize)]);
            if start < 0i64 || end < start || end > i64::cast_from(nnz) {
                status[0].store(1u32);
            } else {
                if end - start > dim { status[0].store(1u32); }
                if start < end {
                    let base = batch * nnz;
                    let mut j = usize::cast_from(start) + 1usize;
                    while j < usize::cast_from(end) {
                        let previous = indices[address(indices, base + j - 1usize)];
                        let current = indices[address(indices, base + j)];
                        if previous >= current { status[0].store(1u32); }
                        j += 1usize;
                    }
                }
            }
        }
    }
}

fn grid(work: usize) -> RudaCount {
    let blocks = work.div_ceil(256); assert!(blocks > 0 && blocks <= u32::MAX as usize);
    let x = blocks.min(131072); RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1)
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64]) {
    let c = client(); let block = RudaDim::new_1d(256);
    if op == 518 {
        assert_eq!(xs.len(), 3); assert_eq!(ints.len(), 3); assert!(ints.iter().all(|&n| n >= 0));
        let (pointers,indices,status) = (&xs[0],&xs[1],&xs[2]);
        assert!(pointers.dtype == 4 || pointers.dtype == 5); assert_eq!(pointers.dtype, indices.dtype);
        assert!(!pointers.shape.is_empty()); assert_eq!(pointers.shape.len(), indices.shape.len());
        let rank = pointers.shape.len(); let (cdim,dim,nnz) = (ints[0] as usize,ints[1],ints[2] as usize);
        assert!(cdim < u32::MAX as usize && nnz <= u32::MAX as usize);
        assert_eq!(pointers.shape[rank-1], cdim+1); assert_eq!(indices.shape[rank-1], nnz);
        assert_eq!(pointers.shape[..rank-1], indices.shape[..rank-1]);
        assert_eq!(status.dtype,5); assert_eq!(status.len,1);
        if pointers.dtype == 5 { assert!(ints.iter().all(|&n| n <= i32::MAX as i64)); }
        let work = pointers.len.max(indices.len); if work == 0 { return; }
        unsafe {
            if pointers.dtype == 4 {
                validate::launch::<i64,CudaRuntime>(&c,grid(work),block,pointers.arg(),indices.arg(),status.arg(),cdim,dim,nnz);
            } else {
                validate::launch::<i32,CudaRuntime>(&c,grid(work),block,pointers.arg(),indices.arg(),status.arg(),cdim,dim,nnz);
            }
        }
    } else {
        assert_eq!(xs.len(),2); let (input,out) = (&xs[0],&xs[1]);
        assert!((4..=8).contains(&input.dtype)); assert!(out.dtype == 4 || out.dtype == 5);
        let (work,nnz,transpose) = if op == 516 {
            assert!(ints.is_empty()); assert!(input.shape.len() <= 1); assert_eq!(out.shape.len(),1);
            assert!(out.len > 0); (out.len,0,false)
        } else {
            assert_eq!(op,517); assert_eq!(ints.len(),2); assert!(ints[0] > 0);
            assert!(ints[1] == 0 || ints[1] == 1); assert!(!input.shape.is_empty());
            let rank = input.shape.len(); assert!(input.shape[rank-1] > 1);
            assert_eq!(out.shape.len(),2); assert_eq!(out.shape[0],rank+1);
            let nnz = ints[0] as usize;
            let batches = input.len / input.shape[rank-1];
            assert_eq!(out.shape[1],batches.checked_mul(nnz).expect("compressed batch size overflow"));
            (out.shape[1],nnz,ints[1] != 0)
        };
        if work == 0 { return; }
        macro_rules! launch {
            ($i:ty, $o:ty) => { unsafe {
                if op == 516 {
                    coo_to_csr::launch::<$i,$o,CudaRuntime>(&c,grid(work),block,input.arg(),out.arg());
                } else {
                    csr_to_coo::launch::<$i,$o,CudaRuntime>(&c,grid(work),block,input.arg(),out.arg(),nnz,transpose);
                }
            }};
        }
        macro_rules! output { ($i:ty) => { if out.dtype == 4 {launch!($i,i64);} else {launch!($i,i32);} }; }
        match input.dtype { 4=>output!(i64), 5=>output!(i32), 6=>output!(i16), 7=>output!(i8), 8=>output!(u8), _=>unreachable!() }
    }
    LAUNCHES.fetch_add(1,Ordering::Relaxed); finish_dispatch(&c);
}
