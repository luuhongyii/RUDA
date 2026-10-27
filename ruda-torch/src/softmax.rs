//! Softmax and log-softmax with storage-aware F32 accumulation.
use super::{bf16, f16, client, finish_dispatch, kernels, CudaRuntime, View, LAUNCHES,
    WARP_SOFTMAX_CALLS, SCALAR_SOFTMAX_CALLS, Ordering};
use ruda_kernel::dsl::prelude::*;

#[ruda]
fn wide_offset(tensor:&Tensor<f64>,row:usize,#[comptime] axis:usize)->usize {
    let mut remaining=row;let mut offset=0usize;let mut d=tensor.rank();
    while d>0 {d-=1;if d!=axis {offset+=(remaining%tensor.shape(d))*tensor.stride(d);remaining/=tensor.shape(d);}}
    offset
}

#[ruda(launch)]
fn wide_softmax(a:&Tensor<f64>,b:&Tensor<f64>,out:&mut Tensor<f64>,#[comptime] axis:usize,#[comptime] backward:bool,#[comptime] logarithmic:bool) {
    let row=RUDA_POS_X as usize;let lane=UNIT_POS as usize;let width=a.shape(axis);
    let ab=wide_offset(a,row,axis);let bb=wide_offset(b,row,axis);let ob=wide_offset(out,row,axis);
    let mut shared=SharedMemory::<f64>::new(256usize);let mut maximum=f64::cast_from(f64::NEG_INFINITY);
    if comptime!(!backward) {
        let mut i=lane;
        while i<width {let x=a[ab+i*a.stride(axis)];if x>maximum || x!=x {maximum=x;}i+=256;}
        shared[lane]=maximum;sync_ruda();let mut stride=128usize;
        while stride>0 {if lane<stride {let other=shared[lane+stride];if other>shared[lane] || other!=other {shared[lane]=other;}}sync_ruda();stride/=2;}
        maximum=shared[0];sync_ruda();
    }
    let mut sum=0.0f64;let mut i=lane;
    while i<width {
        let x=a[ab+i*a.stride(axis)];
        if comptime!(backward) {if comptime!(logarithmic) {sum+=x;}else {sum+=x*b[bb+i*b.stride(axis)];}}
        else {sum+=(x-maximum).exp();}i+=256;
    }
    shared[lane]=sum;sync_ruda();let mut stride=128usize;
    while stride>0 {if lane<stride {shared[lane]=shared[lane]+shared[lane+stride];}sync_ruda();stride/=2;}
    let total=shared[0];let mut i=lane;
    while i<width {
        let x=a[ab+i*a.stride(axis)];let mut y=0.0f64;
        if comptime!(backward) {let value=b[bb+i*b.stride(axis)];if comptime!(logarithmic) {y=x-value.exp()*total;}else {y=value*(x-total);}}
        else {let shifted=x-maximum;if comptime!(logarithmic) {y=shifted-total.ln();}else {y=shifted.exp()/total;}}
        out[ob+i*out.stride(axis)]=y;i+=256;
    }
}

pub(super) fn wide_launch(op:u32,a:&View,b:&View,out:&View,axis:usize) {
    assert!(axis<a.shape.len());assert_eq!(a.dtype,9);assert_eq!(b.dtype,9);assert_eq!(out.dtype,9);
    assert_eq!(a.shape,b.shape);assert_eq!(a.shape,out.shape);if out.len==0 {return;}
    let c=client();let rows=out.len/a.shape[axis];
    unsafe {wide_softmax::launch::<CudaRuntime>(&c,RudaCount::Static(u32::try_from(rows).expect("softmax grid overflow"),1,1),RudaDim::new_1d(256),a.arg(),b.arg(),out.arg(),axis,op>=307,op==306 || op==308);}
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);
}

fn use_warp(width: usize) -> bool {
    match std::env::var("RUDA_TORCH_SOFTMAX") {
        Ok(value) => match value.as_str() {
            "auto" => width >= 32,
            "warp" => true,
            "scalar" => false,
            _ => panic!("RUDA_TORCH_SOFTMAX must be auto, warp, or scalar"),
        },
        Err(std::env::VarError::NotPresent) => width >= 32,
        Err(_) => panic!("RUDA_TORCH_SOFTMAX must be Unicode"),
    }
}

pub(super) fn launch(op: u32, a: &View, b: &View, out: &View, axis: usize) {
    assert!((31..=34).contains(&op));
    assert_eq!(a.dtype, b.dtype);
    assert_eq!(a.shape, out.shape);
    assert_eq!(a.shape, b.shape);
    assert!(axis < a.shape.len());
    if out.len == 0 { return; }
    let rows = out.len / a.shape[axis];
    let warp = use_warp(a.shape[axis]);
    let limit = if warp { u32::MAX as usize / 32 } else { u32::MAX as usize };
    assert!(rows <= limit, "softmax row grid exceeds 32-bit indexing");
    let blocks = if warp { rows.div_ceil(4) } else { rows.div_ceil(128) };
    let count = u32::try_from(blocks).expect("softmax launch grid overflow");
    let client = client();
    let width = a.shape[axis];
    let backward = matches!(op, 32 | 34);
    let cached = axis + 1 == a.shape.len()
        && super::row_cached::eligible(width) && super::row_cached::packed(a)
        && (!backward || super::row_cached::packed(b))
        && super::row_cached::packed(out)
        && std::env::var("RUDA_TORCH_SOFTMAX").map_or(true, |value| value == "auto");
    let threads = if width <= 1024 { 32 } else { super::row_cached::threads(width) };
    let vector = super::row_cached::vector_size(width, a, out)
        .min(if backward { super::row_cached::vector_size(width, b, out) } else { 4 });
    macro_rules! run {
        ($i:ty, $o:ty) => {
            if cached && backward {
                super::row_cached::softmax_backward::launch::<$i, $o, CudaRuntime>(
                    &client, RudaCount::Static(u32::try_from(if threads == 32 { rows.div_ceil(4) } else { rows }).expect("softmax grid overflow"), 1, 1),
                    RudaDim::new_1d(threads.max(128) as u32), a.arg(), b.arg(), out.arg(), op == 34, width, threads, vector)
            } else if cached {
                super::row_cached::softmax::launch::<$i, $o, CudaRuntime>(
                    &client, RudaCount::Static(u32::try_from(if threads == 32 { rows.div_ceil(4) } else { rows }).expect("softmax grid overflow"), 1, 1),
                    RudaDim::new_1d(threads.max(128) as u32), a.arg(), out.arg(), op == 33, width, threads,
                    vector)
            } else if warp {
                kernels::softmax_warp::launch::<$i, $o, CudaRuntime>(
                    &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128),
                    a.arg(), b.arg(), out.arg(), axis, op == 32 || op == 34, op >= 33)
            } else {
                kernels::softmax::launch::<$i, $o, CudaRuntime>(
                    &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128),
                    a.arg(), b.arg(), out.arg(), axis, op == 32 || op == 34, op >= 33)
            }
        };
    }
    unsafe {
        match (a.dtype, out.dtype) {
            (0, 0) => run!(f32, f32),
            (0, 1) => run!(f32, f16),
            (0, 2) => run!(f32, bf16),
            (1, 0) => run!(f16, f32),
            (1, 1) => run!(f16, f16),
            (1, 2) => run!(f16, bf16),
            (2, 0) => run!(bf16, f32),
            (2, 1) => run!(bf16, f16),
            (2, 2) => run!(bf16, bf16),
            _ => panic!("unsupported RUDA softmax dtype combination"),
        }
    }
    finish_dispatch(&client);
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    if warp { WARP_SOFTMAX_CALLS.fetch_add(1, Ordering::Relaxed); }
    else { SCALAR_SOFTMAX_CALLS.fetch_add(1, Ordering::Relaxed); }
}
