use ruda_core::tensor::{DType, Shape};
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_contiguous_dtype, permutation::swap_dims};

// Split K keeps low-rank FP32 projections parallel even with a tiny M or N.
// No TF32 conversion or floating-point atomic accumulation is used.
pub(super) fn try_launch<R: Runtime>(a: &RudaTensor<R>, b: &RudaTensor<R>, out: &RudaTensor<R>) -> bool {
    if a.dtype != DType::F32 || b.dtype != DType::F32 || out.dtype != DType::F32
        || a.qparams.is_some() || b.qparams.is_some() || out.qparams.is_some()
        || a.meta.num_dims() != 2 || b.meta.num_dims() != 2 || out.meta.num_dims() != 2
        || !a.client.same_execution_queue(&b.client) || !a.client.same_execution_queue(&out.client)
    { return false; }
    let (m, k, n) = (a.meta.shape()[0], a.meta.shape()[1], b.meta.shape()[1]);
    if m > 0 && m <= 32 && n > 32 && k >= 512 && b.meta.shape()[0] == k
        && out.meta.shape()[..] == [m, n]
    {
        return try_launch(&swap_dims(b.clone(), 0, 1), &swap_dims(a.clone(), 0, 1),
                          &swap_dims(out.clone(), 0, 1));
    }
    let hw = &a.client.properties().hardware;
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
        columns::launch::<R>(&a.client, grid, RudaDim::new_1d(128),
            a.clone().into_tensor_arg(), b.clone().into_tensor_arg(), partial.clone().into_array_arg(),
            m as u32, k as u32, n as u32, splits as u32);
    } else {
        rows::launch::<R>(&a.client, grid, RudaDim::new_1d(128),
            a.clone().into_tensor_arg(), b.clone().into_tensor_arg(), partial.clone().into_array_arg(),
            m as u32, k as u32, n as u32, splits as u32, n.next_power_of_two());
    }
    reduce::launch::<R>(&a.client, RudaCount::Static((m * n).div_ceil(128) as u32, 1, 1),
        RudaDim::new_1d(128), partial.into_array_arg(), out.clone().into_tensor_arg(), splits as u32);
    true
}

#[ruda(launch)]
fn columns(a: &Tensor<f32>, b: &Tensor<f32>, partial: &mut Array<f32>,
    m: u32, k: u32, n: u32, splits: u32) {
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
                acc = fma(a[row * a.stride(0) + inner * a.stride(1)], b[inner + col * b.stride(1)], acc);
            }
        }
        let total = plane_sum(acc);
        if lane == 0 { partial[split * m as usize * n as usize + output] = total; }
    }
}

#[ruda(launch)]
fn rows(a: &Tensor<f32>, b: &Tensor<f32>, partial: &mut Array<f32>,
    m: u32, k: u32, n: u32, splits: u32, #[comptime] cols: usize) {
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
                acc = fma(a[row * a.stride(0) + inner * a.stride(1)], b[inner * b.stride(0) + col], acc);
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
