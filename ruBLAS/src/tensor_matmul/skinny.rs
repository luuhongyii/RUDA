use ruda_core::tensor::{DType, Shape};
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_contiguous_dtype, permutation::swap_dims};

// Split K keeps low-rank FP32 projections parallel even with a tiny M or N.
// No TF32 conversion or floating-point atomic accumulation is used.
#[allow(unsafe_code)]
pub(super) fn try_launch<R: Runtime>(a: &RudaTensor<R>, b: &RudaTensor<R>, out: &RudaTensor<R>) -> bool {
    if a.dtype != DType::F32 || b.dtype != DType::F32 || out.dtype != DType::F32
        || a.qparams.is_some() || b.qparams.is_some() || out.qparams.is_some()
        || a.meta.num_dims() != 2 || b.meta.num_dims() != 2 || out.meta.num_dims() != 2
        || !a.client.same_execution_queue(&b.client) || !a.client.same_execution_queue(&out.client)
    { return false; }
    let (m, k, n) = (a.meta.shape()[0], a.meta.shape()[1], b.meta.shape()[1]);
    let hw = &a.client.properties().hardware;
    if k == 16 && m > 0 && m % 16 == 0 && n >= 128 && n % 128 == 0
        && a.meta.strides()[..] == [16, 1] && b.meta.shape()[0] == 16
        && b.meta.strides()[..] == [1, 16] && out.meta.shape()[..] == [m, n]
        && out.meta.strides()[..] == [n, 1]
        && b.handle.offset_start.unwrap_or(0) % 16 == 0
        && m.checked_mul(n).is_some_and(|v| v <= u32::MAX as usize)
        && n <= u32::MAX as usize / 16 && m <= u32::MAX as usize / 16
        && hw.plane_size_min == 32 && hw.plane_size_max == 32
        && n / 128 <= hw.max_ruda_count.0 as usize && m / 16 <= hw.max_ruda_count.1 as usize
    {
        unsafe { rank16::launch_unchecked::<R>(&a.client, RudaCount::Static((n / 128) as u32, (m / 16) as u32, 1),
            RudaDim::new_1d(128), a.clone().into_array_arg(), b.clone().into_array_arg(),
            out.clone().into_array_arg(), n); }
        return true;
    }
    if m > 0 && m <= 32 && n > 32 && k >= 512 && b.meta.shape()[0] == k
        && out.meta.shape()[..] == [m, n]
    {
        return try_launch(&swap_dims(b.clone(), 0, 1), &swap_dims(a.clone(), 0, 1),
                          &swap_dims(out.clone(), 0, 1));
    }
    if m == 0 || k < 512 || n == 0 || n > 32 || b.meta.shape()[0] != k
        || out.meta.shape()[..] != [m, n] || (a.meta.strides()[1] != 1 && a.meta.strides()[0] != 1)
        || !((out.meta.strides()[1] == 1 && (m == 1 || out.meta.strides()[0] == n))
            || (out.meta.strides()[0] == 1 && (n == 1 || out.meta.strides()[1] == m)))
        || hw.plane_size_min != 32 || hw.plane_size_max != 32
    { return false; }
    let column_major = b.meta.strides()[0] == 1;
    if !column_major && b.meta.strides()[1] != 1 { return false; }
    for tensor in [a, b, out] {
        let span = tensor.meta.shape().iter().zip(tensor.meta.strides().iter()).try_fold(1usize,
            |sum, (&dim, &stride)| sum.checked_add((dim - 1).checked_mul(stride)?));
        if span.is_none_or(|span| span > u32::MAX as usize) { return false; }
    }
    let tile = if column_major { 1024 } else { 512 };
    let splits = k.div_ceil(tile);
    let Some(count) = m.checked_mul(n).and_then(|v| v.checked_mul(splits)) else { return false; };
    if count > 16 * 1024 * 1024 { return false; }
    let warps = if column_major { count } else { m * splits };
    if warps.div_ceil(4) > hw.max_ruda_count.0 as usize { return false; }
    let partial = empty_device_contiguous_dtype(a.client.clone(), a.device.clone(), Shape::from([count]), DType::F32);
    let grid = RudaCount::Static(warps.div_ceil(4) as u32, 1, 1);
    if column_major {
        unsafe { columns::launch_unchecked::<R>(&a.client, grid, RudaDim::new_1d(128),
            a.clone().into_tensor_arg(), b.clone().into_tensor_arg(), partial.clone().into_array_arg(),
            m, k, n, splits, a.meta.strides()[0], a.meta.strides()[1], b.meta.strides()[1]); }
    } else {
        unsafe { rows::launch_unchecked::<R>(&a.client, grid, RudaDim::new_1d(128),
            a.clone().into_tensor_arg(), b.clone().into_tensor_arg(), partial.clone().into_array_arg(),
            m, k, n, splits, n.next_power_of_two(), a.meta.strides()[0], a.meta.strides()[1], b.meta.strides()[0]); }
    }
    reduce::launch::<R>(&a.client, RudaCount::Static((m * n).div_ceil(128) as u32, 1, 1),
        RudaDim::new_1d(128), partial.into_array_arg(), out.clone().into_tensor_arg(), splits as u32);
    true
}

#[ruda(launch_unchecked)]
fn columns(a: &Tensor<f32>, b: &Tensor<f32>, partial: &mut Array<f32>,
    #[comptime] m: usize, #[comptime] k: usize, #[comptime] n: usize, #[comptime] splits: usize,
    #[comptime] as0: usize, #[comptime] as1: usize, #[comptime] bs1: usize) {
    let warp = RUDA_POS_X as usize * 4 + UNIT_POS as usize / 32;
    let lane = UNIT_POS as usize % 32;
    let output = warp / splits as usize;
    let split = warp % splits as usize;
    if output < m as usize * n as usize {
        let row = output / n as usize;
        let col = output % n as usize;
        let mut acc = 0.0f32;
        for offset in 0usize..32 {
            let inner = split * 1024 + offset * 32 + lane;
            if inner < k as usize {
                acc = fma(a[row * as0 + inner * as1], b[inner + col * bs1], acc);
            }
        }
        let total = plane_sum(acc);
        if lane == 0 { partial[split * m as usize * n as usize + output] = total; }
    }
}

#[ruda(launch_unchecked)]
fn rows(a: &Tensor<f32>, b: &Tensor<f32>, partial: &mut Array<f32>,
    #[comptime] m: usize, #[comptime] k: usize, #[comptime] n: usize, #[comptime] splits: usize,
    #[comptime] cols: usize, #[comptime] as0: usize, #[comptime] as1: usize, #[comptime] bs0: usize) {
    let warp = RUDA_POS_X as usize * 4 + UNIT_POS as usize / 32;
    let lane = UNIT_POS as usize % 32;
    let row = warp / splits as usize;
    let split = warp % splits as usize;
    let col = lane % cols;
    if row < m as usize {
        let mut acc = 0.0f32;
        for offset in 0usize..(512 / (32 / cols)) {
            let inner = split * 512 + offset * (32 / cols) + lane / cols;
            if inner < k as usize && col < n as usize {
                acc = fma(a[row * as0 + inner * as1], b[inner * bs0 + col], acc);
            }
        }
        if comptime!(cols <= 16) { acc += plane_shuffle_xor(acc, 16u32); }
        if comptime!(cols <= 8) { acc += plane_shuffle_xor(acc, 8u32); }
        if comptime!(cols <= 4) { acc += plane_shuffle_xor(acc, 4u32); }
        if comptime!(cols <= 2) { acc += plane_shuffle_xor(acc, 2u32); }
        if comptime!(cols == 1) { acc += plane_shuffle_xor(acc, 1u32); }
        if lane < cols && col < n as usize {
            partial[(split * m as usize + row) * n as usize + col] = acc;
        }
    }
}

#[ruda(launch_unchecked)]
fn rank16(a: &Array<f32>, b: &Array<Vector<f32, Const<4>>>, out: &mut Array<f32>,
    #[comptime] n: usize) {
    let thread = UNIT_POS as usize;
    let row = RUDA_POS_Y as usize * 16;
    let col = RUDA_POS_X as usize * 128 + thread;
    let mut shared = SharedMemory::<f32>::new(256usize);
    #[unroll]
    for i in 0usize..2 { shared[thread + i * 128] = a[row * 16 + thread + i * 128]; }
    sync_ruda();
    let mut acc = Array::<f32>::new(16usize);
    #[unroll]
    for i in 0usize..16 { acc[i] = 0.0f32; }
    #[unroll]
    for group in 0usize..4 {
        let value = b[col * 4 + group];
        #[unroll]
        for k in 0usize..4 {
            #[unroll]
            for i in 0usize..16 { acc[i] = fma(shared[i * 16 + group * 4 + k], value[k], acc[i]); }
        }
    }
    #[unroll]
    for i in 0usize..16 { out[(row + i) * n + col] = acc[i]; }
}

#[ruda(launch)]
fn reduce(partial: &Array<f32>, out: &mut Tensor<f32>, splits: u32) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let mut sum = 0.0f32;
        for split in 0usize..splits as usize { sum += partial[split * out.len() + pos]; }
        let row = pos / out.shape(1);
        let col = pos % out.shape(1);
        out[row * out.stride(0) + col * out.stride(1)] = sum;
    }
}
