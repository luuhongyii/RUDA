use ruda_kernel::dsl::prelude::*;

#[ruda]
fn block_reduce(value: f32, scratch: &mut SharedMemory<f32>,
    #[comptime] threads: usize, #[comptime] maximum: bool,
) -> f32 {
    let thread = UNIT_POS as usize;
    let mut reduced = 0.0f32;
    if comptime!(maximum) {
        let nan = plane_max(f32::cast_from(value != value));
        reduced = plane_max(value);
        if nan > 0.0 { reduced = f32::new(f32::NAN); }
    } else { reduced = plane_sum(value); }
    if thread % 32 == 0 { scratch[thread / 32] = reduced; }
    sync_ruda();
    if thread < 32 {
        let mut partial = 0.0f32;
        if comptime!(maximum) { partial = f32::new(f32::NEG_INFINITY); }
        if thread < threads / 32 { partial = scratch[thread]; }
        if comptime!(maximum) {
            let nan = plane_max(f32::cast_from(partial != partial));
            reduced = plane_max(partial);
            if nan > 0.0 { reduced = f32::new(f32::NAN); }
        } else { reduced = plane_sum(partial); }
        if thread == 0 { scratch[0] = reduced; }
    }
    sync_ruda();
    let result = scratch[0];
    sync_ruda();
    result
}

#[ruda(launch)]
pub fn norm<F: Float + RudaElement, W: Float + RudaElement, B: Float + RudaElement, S: Float + RudaElement>(
    input: &Tensor<F>, weight: &Tensor<W>, bias: &Tensor<B>, out: &mut Tensor<F>,
    mean: &mut Tensor<S>, rstd: &mut Tensor<S>, epsilon: f32,
    #[comptime] weighted: bool, #[comptime] biased: bool, #[comptime] layer: bool,
    #[comptime] statistics: bool,
    #[comptime] width: usize, #[comptime] threads: usize,
) {
    let row = RUDA_POS_X as usize;
    let thread = UNIT_POS as usize;
    let mut values = Array::<f32>::new((width + threads - 1) / threads);
    let mut scratch = SharedMemory::<f32>::new(threads / 32);
    let mut local = 0.0f32;
    #[unroll]
    for j in 0usize..((width + threads - 1) / threads) {
        let col = thread + j * threads;
        let mut value = 0.0f32;
        if col < width { value = f32::cast_from(input[row * width + col]); }
        values[j] = value;
        if comptime!(layer) { local += value; } else { local += value * value; }
    }
    let total = block_reduce(local, &mut scratch, threads, false);
    let mut mu = 0.0f32;
    let mut variance = total / width as f32;
    if comptime!(layer) {
        mu = variance;
        local = 0.0f32;
        #[unroll]
        for j in 0usize..((width + threads - 1) / threads) {
            let col = thread + j * threads;
            if col < width {
                let centered = values[j] - mu;
                values[j] = centered;
                local += centered * centered;
            }
        }
        variance = block_reduce(local, &mut scratch, threads, false) / width as f32;
    }
    let inv = (variance + epsilon).inverse_sqrt();
    if comptime!(statistics) {
        if thread == 0 {
            if comptime!(layer) { mean[row] = S::cast_from(mu); }
            rstd[row] = S::cast_from(inv);
        }
    }
    #[unroll]
    for j in 0usize..((width + threads - 1) / threads) {
        let col = thread + j * threads;
        if col < width {
            let mut value = values[j] * inv;
            if comptime!(weighted) { value *= f32::cast_from(weight[col]); }
            if comptime!(biased) { value += f32::cast_from(bias[col]); }
            out[row * width + col] = F::cast_from(value);
        }
    }
}

#[ruda(launch)]
pub fn softmax<F: Float + RudaElement, O: Float + RudaElement>(
    input: &Tensor<F>, out: &mut Tensor<O>, #[comptime] logarithmic: bool,
    #[comptime] width: usize, #[comptime] threads: usize,
) {
    let row = RUDA_POS_X as usize;
    let thread = UNIT_POS as usize;
    let mut values = Array::<f32>::new((width + threads - 1) / threads);
    let mut scratch = SharedMemory::<f32>::new(threads / 32);
    let mut local = f32::new(f32::NEG_INFINITY);
    #[unroll]
    for j in 0usize..((width + threads - 1) / threads) {
        let col = thread + j * threads;
        let mut value = f32::new(f32::NEG_INFINITY);
        if col < width { value = f32::cast_from(input[row * width + col]); }
        values[j] = value;
        if value > local || value != value { local = value; }
    }
    let maximum = block_reduce(local, &mut scratch, threads, true);
    local = 0.0f32;
    #[unroll]
    for j in 0usize..((width + threads - 1) / threads) {
        let col = thread + j * threads;
        if col < width {
            let shifted = values[j] - maximum;
            let exponential = shifted.exp();
            local += exponential;
            if comptime!(logarithmic) { values[j] = shifted; } else { values[j] = exponential; }
        }
    }
    let total = block_reduce(local, &mut scratch, threads, false);
    let mut log_total = 0.0f32;
    if comptime!(logarithmic) { log_total = total.ln(); }
    #[unroll]
    for j in 0usize..((width + threads - 1) / threads) {
        let col = thread + j * threads;
        if col < width {
            let mut value = values[j];
            if comptime!(logarithmic) { value -= log_total; } else { value /= total; }
            out[row * width + col] = O::cast_from(value);
        }
    }
}

pub(super) fn threads(width: usize) -> usize { if width <= 4096 { 128 } else { 256 } }

pub(super) fn eligible(width: usize) -> bool { (512..=16384).contains(&width) }

pub(super) fn packed(view: &super::View) -> bool {
    let mut stride = 1usize;
    for (&dim, &actual) in view.shape.iter().zip(&view.strides).rev() {
        if dim > 1 && actual != stride { return false; }
        stride = stride.checked_mul(dim).expect("row stride overflow");
    }
    true
}
