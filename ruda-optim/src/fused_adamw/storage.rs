// SPDX-License-Identifier: Apache-2.0
//! Low-level in-place storage kernels for native framework adaptors.
//!
//! Unlike `adamw_step`, these preserve externally owned parameter/state addresses.
//! The caller MUST validate contiguous shapes, distinct writable buffers, FP32
//! moments/master weights, launch sizes, and the entire group's finite-gradient
//! policy before updating any parameter. These generated launch APIs are unsafe.
//! No CPU fallback, global optimizer state, autograd registration or implicit
//! model casting is installed here. Existing out-of-place/AMSGrad/clipping APIs
//! and their checkpoint formats are unchanged.
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

// Each warp owns one flag and visits a strided part of the gradient. At most
// 1024 flags per parameter, independent of parameter size. Check both stored
// input and unscaled output; NaN/Inf are not replaced with fabricated values.
#[ruda(launch)]
pub fn unscale_check<F: Float + RudaElement>(
    grad: &mut Tensor<F>, flags: &mut Tensor<f32>, inv_scale: f32,
) {
    let warp = (ABSOLUTE_POS / 32) as usize;
    let lane = (ABSOLUTE_POS % 32) as usize;
    if warp < flags.len() {
        let mut pos = ABSOLUTE_POS as usize;
        let mut bad = 0.0f32;
        while pos < grad.len() {
            let value = f32::cast_from(grad[pos]);
            let result = F::cast_from(value * inv_scale);
            let stored = f32::cast_from(result);
            if value != value || value.abs() > 3.4028234663852886e38f32
                || stored != stored || stored.abs() > 3.4028234663852886e38f32 {
                bad = 1.0f32;
            }
            grad[pos] = result;
            pos += flags.len() * 32;
        }
        let found = plane_max(bad);
        if lane == 0 { flags[warp] = found; }
    }
}

#[ruda(launch)]
pub fn merge_flags(flags: &Tensor<f32>, found: &mut Tensor<f32>) {
    let lane = UNIT_POS as usize;
    let mut pos = lane;
    let mut bad = 0.0f32;
    while pos < flags.len() {
        bad = f32::max(bad, flags[pos]);
        pos += 32;
    }
    let result = plane_max(bad);
    if lane == 0 { found[0] = result; }
}

#[ruda(launch)]
pub fn adamw<F: Float + RudaElement>(
    parameter: &mut Tensor<F>, gradient: &Tensor<F>, master: &mut Tensor<f32>,
    first: &mut Tensor<f32>, second: &mut Tensor<f32>,
    lr: f32, beta1: f32, beta2: f32, epsilon: f32, decay: f32,
    correction1: f32, correction2: f32, #[comptime] separate_master: bool,
) {
    let pos = ABSOLUTE_POS as usize;
    if pos < parameter.len() {
        let g = f32::cast_from(gradient[pos]);
        let m = beta1 * first[pos] + (1.0f32 - beta1) * g;
        let v = beta2 * second[pos] + (1.0f32 - beta2) * g * g;
        let mut p = f32::cast_from(parameter[pos]);
        if comptime!(separate_master) { p = master[pos]; }
        p = p * (1.0f32 - lr * decay)
            - lr * (m / correction1) / ((v / correction2).sqrt() + epsilon);
        first[pos] = m;
        second[pos] = v;
        if comptime!(separate_master) { master[pos] = p; }
        parameter[pos] = F::cast_from(p);
    }
}


/// Read-only finite check and optional stable L2 statistics of FP32-unscaled
/// gradients. Each warp writes (maximum magnitude, scaled sum of squares, bad).
/// No gradient writeback, no low-precision unscale rounding, and no global atomics.
#[ruda(launch)]
pub fn analyze_gradient<F: Float + RudaElement>(
    grad: &Tensor<F>, stats: &mut Tensor<f32>, inv_scale: f32,
    #[comptime] with_norm: bool,
) {
    analyze_gradient_at::<F>(grad, stats, inv_scale, ABSOLUTE_POS as usize, with_norm);
}

#[ruda]
fn analyze_gradient_at<F: Float + RudaElement>(
    grad: &Tensor<F>, stats: &mut Tensor<f32>, inv_scale: f32, position: usize,
    #[comptime] with_norm: bool,
) {
    let warp = position / 32;
    let lane = position % 32;
    let warps = stats.len() / 3;
    if warp < warps {
        let mut pos = position;
        let mut bad = 0.0f32;
        let mut scale = 0.0f32;
        let mut sum = 0.0f32;
        while pos < grad.len() {
            let raw = f32::cast_from(grad[pos]);
            let value = raw * inv_scale;
            if raw != raw || raw.abs() > 3.4028234663852886e38f32
                || value != value || value.abs() > 3.4028234663852886e38f32 {
                bad = 1.0f32;
            } else if comptime!(with_norm) {
                let a = value.abs();
                if a > scale {
                    let ratio = scale / a;
                    sum = 1.0f32 + sum * ratio * ratio;
                    scale = a;
                } else if a > 0.0f32 {
                    let ratio = a / scale;
                    sum += ratio * ratio;
                }
            }
            pos += warps * 32;
        }
        let largest = plane_max(scale);
        let mut contribution = 0.0f32;
        if largest > 0.0f32 {
            let ratio = scale / largest;
            contribution = sum * ratio * ratio;
        }
        let squares = plane_sum(contribution);
        let invalid = plane_max(bad);
        if lane == 0 {
            stats[warp * 3] = largest;
            stats[warp * 3 + 1] = squares;
            stats[warp * 3 + 2] = invalid;
        }
    }
}

#[ruda(launch)]
pub fn analyze_gradients<F: Float + RudaElement>(
    gradients: &Sequence<Tensor<F>>, stats: &mut Sequence<Tensor<f32>>, inv_scale: f32,
    #[comptime] with_norm: bool,
) {
    let block = RUDA_POS_X as usize;
    let mut first = 0usize;
    #[unroll]
    for tensor in 0..gradients.len() {
        let output = stats.index_mut(tensor);
        let warps = output.len() / 3;
        let blocks = warps / 4 + usize::cast_from(warps % 4 != 0);
        if block >= first && block < first + blocks {
            analyze_gradient_at::<F>(gradients.index(tensor), output, inv_scale,
                (block - first) * 128 + UNIT_POS as usize, with_norm);
        }
        first += blocks;
    }
}

/// Report is (bad, maximum magnitude, scaled sum of squares). Keeping a scaled
/// representation avoids overflowing a finite norm when gradients are large.
/// Exactly one warp merges the bounded partial workspace; no signal goes to CPU.
#[ruda(launch)]
pub fn merge_gradient_stats(stats: &Tensor<f32>, report: &mut Tensor<f32>) {
    let lane = UNIT_POS as usize;
    let rows = stats.len() / 3;
    let mut row = lane;
    let mut largest = 0.0f32;
    let mut invalid = 0.0f32;
    while row < rows {
        largest = f32::max(largest, stats[row * 3]);
        invalid = f32::max(invalid, stats[row * 3 + 2]);
        row += 32;
    }
    let scale = plane_max(largest);
    let bad = plane_max(invalid);
    let mut sum = 0.0f32;
    row = lane;
    if scale > 0.0f32 {
        while row < rows {
            let ratio = stats[row * 3] / scale;
            sum += stats[row * 3 + 1] * ratio * ratio;
            row += 32;
        }
    }
    let squares = plane_sum(sum);
    if lane == 0 {
        report[0] = bad;
        report[1] = scale;
        report[2] = squares;
    }
}

/// Fused scale/clip/update; gradient storage stays unchanged. Apply the factors
/// separately to avoid underflow from pre-multiplying two small scale factors.
#[ruda(launch)]
pub fn adamw_scaled<F: Float + RudaElement>(
    parameter: &mut Tensor<F>, gradient: &Tensor<F>, master: &mut Tensor<f32>,
    first: &mut Tensor<f32>, second: &mut Tensor<f32>,
    lr: f32, beta1: f32, beta2: f32, epsilon: f32, decay: f32,
    correction1: f32, correction2: f32, inv_scale: f32, clip: f32,
    #[comptime] separate_master: bool,
) {
    adamw_scaled_at::<F>(parameter, gradient, master, first, second, lr, beta1, beta2, epsilon,
        decay, correction1, correction2, inv_scale, clip, ABSOLUTE_POS as usize, separate_master);
}

#[ruda]
fn adamw_scaled_at<F: Float + RudaElement>(
    parameter: &mut Tensor<F>, gradient: &Tensor<F>, master: &mut Tensor<f32>,
    first: &mut Tensor<f32>, second: &mut Tensor<f32>,
    lr: f32, beta1: f32, beta2: f32, epsilon: f32, decay: f32,
    correction1: f32, correction2: f32, inv_scale: f32, clip: f32, pos: usize,
    #[comptime] separate_master: bool,
) {
    if pos < parameter.len() {
        let g = (f32::cast_from(gradient[pos]) * inv_scale) * clip;
        let m = beta1 * first[pos] + (1.0f32 - beta1) * g;
        let v = beta2 * second[pos] + (1.0f32 - beta2) * g * g;
        let mut p = f32::cast_from(parameter[pos]);
        if comptime!(separate_master) { p = master[pos]; }
        p = p * (1.0f32 - lr * decay)
            - lr * (m / correction1) / ((v / correction2).sqrt() + epsilon);
        first[pos] = m;
        second[pos] = v;
        if comptime!(separate_master) { master[pos] = p; }
        parameter[pos] = F::cast_from(p);
    }
}

#[ruda(launch)]
pub fn adamw_scaled_batch<F: Float + RudaElement>(
    parameters: &mut Sequence<Tensor<F>>, gradients: &Sequence<Tensor<F>>,
    masters: &mut Sequence<Tensor<f32>>, first: &mut Sequence<Tensor<f32>>,
    second: &mut Sequence<Tensor<f32>>, hypers: &Sequence<f32>, inv_scale: f32, clip: f32,
    #[comptime] separate_master: bool,
) {
    let block = RUDA_POS_X as usize;
    let mut start = 0usize;
    #[unroll]
    for tensor in 0..parameters.len() {
        let parameter = parameters.index_mut(tensor);
        let blocks = parameter.len() / 128 + usize::cast_from(parameter.len() % 128 != 0);
        if block >= start && block < start + blocks {
            adamw_scaled_at::<F>(parameter, gradients.index(tensor), masters.index_mut(tensor),
                first.index_mut(tensor), second.index_mut(tensor),
                *hypers.index(tensor * 7), *hypers.index(tensor * 7 + 1),
                *hypers.index(tensor * 7 + 2), *hypers.index(tensor * 7 + 3),
                *hypers.index(tensor * 7 + 4), *hypers.index(tensor * 7 + 5),
                *hypers.index(tensor * 7 + 6), inv_scale, clip,
                (block - start) * 128 + UNIT_POS as usize, separate_master);
        }
        start += blocks;
    }
}


/// Each warp reduces one contiguous chunk of (scale, squares, bad) rows.
/// Output uses the SAME row representation, allowing multiple reduction levels.
/// The final merge_gradient_stats changes only the report field order.
/// Caller must provide disjoint, exact-size input/output views and a positive
/// compile-time fan-in. No atomics, gradient writes, or host tensor transfer.
#[ruda(launch)]
pub fn merge_gradient_stats_chunks(
    stats: &Tensor<f32>, partial: &mut Tensor<f32>, #[comptime] fan_in: usize,
) {
    let warp = (ABSOLUTE_POS / 32) as usize;
    let lane = (ABSOLUTE_POS % 32) as usize;
    if warp < partial.len() / 3 {
        let first = warp * fan_in;
        let end = usize::min(first + fan_in, stats.len() / 3);
        let mut row = first + lane;
        let mut largest = 0.0f32;
        let mut invalid = 0.0f32;
        while row < end {
            largest = f32::max(largest, stats[row * 3]);
            invalid = f32::max(invalid, stats[row * 3 + 2]);
            row += 32;
        }
        let scale = plane_max(largest);
        let bad = plane_max(invalid);
        let mut sum = 0.0f32;
        row = first + lane;
        if scale > 0.0f32 {
            while row < end {
                let ratio = stats[row * 3] / scale;
                sum += stats[row * 3 + 1] * ratio * ratio;
                row += 32;
            }
        }
        let squares = plane_sum(sum);
        if lane == 0 {
            partial[warp * 3] = scale;
            partial[warp * 3 + 1] = squares;
            partial[warp * 3 + 2] = bad;
        }
    }
}
