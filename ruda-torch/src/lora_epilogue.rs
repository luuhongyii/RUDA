//! FP32 adapter scaling followed by low-precision rounding and residual add.
use super::*;
use super::scientific::address;

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_lora_projection_api_version() -> u32 { 1 }

#[ruda(launch_unchecked)]
fn rank16_residual<S: Float + RudaElement>(
    base: &Array<S>, rank: &Array<f32>, weight: &Array<Vector<f32, Const<4>>>,
    output: &mut Array<S>, scale: f32, #[comptime] n: usize,
) {
    let thread = UNIT_POS as usize;
    let row = RUDA_POS_Y as usize * 16;
    let col = RUDA_POS_X as usize * 128 + thread;
    let mut shared = SharedMemory::<f32>::new(256usize);
    #[unroll]
    for i in 0usize..2 { shared[thread + i * 128] = rank[row * 16 + thread + i * 128]; }
    sync_ruda();
    let mut acc = Array::<f32>::new(16usize);
    #[unroll]
    for i in 0usize..16 { acc[i] = 0.0f32; }
    #[unroll]
    for group in 0usize..4 {
        let value = weight[col * 4 + group];
        #[unroll]
        for k in 0usize..4 {
            #[unroll]
            for i in 0usize..16 { acc[i] = fma(shared[i * 16 + group * 4 + k], value[k], acc[i]); }
        }
    }
    #[unroll]
    for i in 0usize..16 {
        let at = (row + i) * n + col;
        let stored = S::cast_from(acc[i] * scale);
        output[at] = S::cast_from(f32::cast_from(base[at]) + f32::cast_from(stored));
    }
}

pub(super) fn projection(xs: &[View], ints: &[i64], values: &[f64]) {
    assert!(ints.is_empty()); assert_eq!(values.len(), 1); assert_eq!(xs.len(), 4);
    let (base, rank, weight, output) = (&xs[0], &xs[1], &xs[2], &xs[3]);
    assert!(matches!(base.dtype, 1 | 2)); assert_eq!(output.dtype, base.dtype);
    assert_eq!(rank.dtype, 0); assert_eq!(weight.dtype, 0);
    assert!(!base.shape.is_empty()); assert_eq!(output.shape, base.shape);
    let n = *base.shape.last().unwrap();
    assert!(n >= 128 && n % 128 == 0);
    let m = base.len / n;
    assert!(m > 0 && m % 16 == 0);
    assert_eq!(rank.shape.len(), base.shape.len());
    assert_eq!(&rank.shape[..rank.shape.len()-1], &base.shape[..base.shape.len()-1]);
    assert_eq!(*rank.shape.last().unwrap(), 16); assert_eq!(weight.shape, [n, 16]);
    for x in xs {
        let mut stride = 1usize;
        for (&size, &actual) in x.shape.iter().zip(&x.strides).rev() {
            assert!(size <= 1 || actual == stride, "LoRA rank projection requires contiguous storage");
            stride = stride.checked_mul(size).expect("LoRA rank projection shape overflow");
        }
        assert!(x.len <= u32::MAX as usize);
    }
    assert_eq!(weight.handle.offset_start.unwrap_or(0) % 16, 0);
    let c = client();
    let hw = &c.properties().hardware;
    assert!(n / 128 <= hw.max_ruda_count.0 as usize && m / 16 <= hw.max_ruda_count.1 as usize);
    let grid = RudaCount::Static((n / 128) as u32, (m / 16) as u32, 1);
    macro_rules! run { ($s:ty) => { unsafe {
        rank16_residual::launch_unchecked::<$s, CudaRuntime>(&c, grid, RudaDim::new_1d(128),
            primitives::tensor(base).into_array_arg(), primitives::tensor(rank).into_array_arg(),
            primitives::tensor(weight).into_array_arg(), primitives::tensor(output).into_array_arg(),
            values[0] as f32, n);
    } }; }
    match base.dtype { 1 => run!(f16), 2 => run!(bf16), _ => unreachable!() }
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
}

#[ruda(launch)]
fn forward<S: Float + RudaElement>(
    base: &Tensor<S>, update: &Tensor<f32>, output: &mut Tensor<S>, scale: f32,
) {
    let pos = ABSOLUTE_POS as usize;
    if pos < output.len() {
        let scaled = update[address(update, pos)] * scale;
        let stored = S::cast_from(scaled);
        output[pos] = S::cast_from(f32::cast_from(base[address(base, pos)]) + f32::cast_from(stored));
    }
}

#[ruda(launch)]
fn backward<S: Float + RudaElement>(grad: &Tensor<S>, output: &mut Tensor<f32>, scale: f32) {
    let pos = ABSOLUTE_POS as usize;
    if pos < output.len() { output[pos] = f32::cast_from(grad[address(grad, pos)]) * scale; }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64], values: &[f64]) {
    assert!(ints.is_empty()); assert_eq!(values.len(), 1);
    assert_eq!(xs.len(), if op == 654 { 3 } else { 2 });
    let base = &xs[0]; let output = xs.last().unwrap();
    assert!(base.dtype == 1 || base.dtype == 2);
    assert!(xs.iter().all(|x| x.shape == base.shape));
    if op == 654 {
        assert_eq!(xs[1].dtype, 0); assert_eq!(output.dtype, base.dtype);
    } else { assert_eq!(output.dtype, 0); }
    let mut stride = 1usize;
    for (&size, &actual) in output.shape.iter().zip(&output.strides).rev() {
        assert!(size <= 1 || actual == stride, "LoRA epilogue output must be contiguous");
        stride = stride.checked_mul(size).expect("LoRA epilogue shape overflow");
    }
    assert!(base.len <= u32::MAX as usize, "LoRA epilogue grid overflow");
    if base.len == 0 { return; }
    let c = client(); let grid = RudaCount::Static(base.len.div_ceil(128) as u32, 1, 1);
    let scale = values[0] as f32;
    macro_rules! run { ($s:ty) => { unsafe {
        if op == 654 {
            forward::launch::<$s,CudaRuntime>(&c,grid,RudaDim::new_1d(128),
                base.arg(),xs[1].arg(),output.arg(),scale);
        } else {
            backward::launch::<$s,CudaRuntime>(&c,grid,RudaDim::new_1d(128),base.arg(),output.arg(),scale);
        }
    } }; }
    match base.dtype {1=>run!(f16),2=>run!(bf16),_=>unreachable!()}
    LAUNCHES.fetch_add(1,Ordering::Relaxed);
}
