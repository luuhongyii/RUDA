use super::*;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn multiply(left: &Tensor<i8>, right: &Tensor<i8>, out: &mut Tensor<i32>,
    column_tiles: usize, inner_tiles: usize, blocks: usize,
) {
    let block = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    if block >= blocks { terminate!(); }
    let tid = UNIT_POS as usize; let tx = tid % 16usize; let ty = tid / 16usize;
    let row = block / column_tiles * 16usize + ty; let column = block % column_tiles * 16usize + tx;
    let rows = left.shape(0); let inner = left.shape(1); let columns = right.shape(1);
    let mut a_tile = SharedMemory::<i32>::new(256usize);
    let mut b_tile = SharedMemory::<i32>::new(256usize);
    let mut sum = 0u32;
    for tile in 0..inner_tiles {
        let ak = tile * 16usize + tx; let bk = tile * 16usize + ty;
        let mut a = 0i32; let mut b = 0i32;
        if row < rows && ak < inner { a = i32::cast_from(left[row * left.stride(0) + ak * left.stride(1)]); }
        if bk < inner && column < columns { b = i32::cast_from(right[bk * right.stride(0) + column * right.stride(1)]); }
        a_tile[tid] = a; b_tile[tid] = b;
        sync_ruda();
        for k in 0..16usize { sum += u32::cast_from(a_tile[ty * 16usize + k] * b_tile[k * 16usize + tx]); }
        sync_ruda();
    }
    if row < rows && column < columns { out[row * columns + column] = i32::cast_from(sum); }
}

pub(super) fn dispatch(xs: &[View]) {
    assert_eq!(xs.len(), 3);
    let (a, b, out) = (&xs[0], &xs[1], &xs[2]);
    assert_eq!(a.dtype, 7); assert_eq!(b.dtype, 7); assert_eq!(out.dtype, 5);
    assert_eq!(a.shape.len(), 2); assert_eq!(b.shape.len(), 2);
    let (m, k, n) = (a.shape[0], a.shape[1], b.shape[1]);
    assert!(m > 16 && k > 0 && k % 8 == 0 && n > 0 && n % 8 == 0);
    assert_eq!(b.shape[0], k); assert_eq!(out.shape, vec![m, n]); assert_eq!(out.strides, vec![n, 1]);
    let columns = n.div_ceil(16); let blocks = m.div_ceil(16).checked_mul(columns).expect("integer matmul grid overflow");
    assert!(blocks <= u32::MAX as usize);
    let c = client(); let x = blocks.min(131072);
    unsafe { multiply::launch::<CudaRuntime>(&c, RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1),
        RudaDim::new_1d(256), a.arg(), b.arg(), out.arg(), columns, k.div_ceil(16), blocks); }
    LAUNCHES.fetch_add(1, Ordering::Relaxed); finish_dispatch(&c);
}
