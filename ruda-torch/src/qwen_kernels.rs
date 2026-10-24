use super::*;

#[ruda(launch)]
pub(super) fn rms_decode<F: Float + RudaElement, N: Size>(
    input: &Array<Vector<F, N>>, weight: &Array<Vector<F, N>>, output: &mut Array<Vector<F, N>>,
    vectors: u32, width: u32, epsilon: f32,
) {
    let lane = UNIT_POS_X;
    let warp = UNIT_POS_Y;
    let mut square = Vector::<f32,N>::new(0.0f32);
    let mut column = warp * 32 + lane;
    while column < vectors {
        let value = Vector::<f32,N>::cast_from(input[column as usize]);
        square += value * value;
        column += 128;
    }
    let sum = f32::cast_from(plane_sum(Vector::vector_sum(square)));
    let mut partial = SharedMemory::<f32>::new(4usize);
    if lane == 0 { partial[warp as usize] = sum; }
    sync_ruda();
    let mut total = 0.0f32;
    #[unroll] for i in 0usize..4 { total += partial[i]; }
    let inverse = Vector::<f32,N>::new((total / width as f32 + epsilon).inverse_sqrt());
    let mut column = warp * 32 + lane;
    while column < vectors {
        let normalized = Vector::<F,N>::cast_from(Vector::<f32,N>::cast_from(input[column as usize]) * inverse);
        output[column as usize] = Vector::<F,N>::cast_from(
            Vector::<f32,N>::cast_from(normalized) * Vector::<f32,N>::cast_from(weight[column as usize]));
        column += 128;
    }
}

#[ruda(launch)]
pub(super) fn linear<F: Float + RudaElement, N: Size>(
    input: &Array<Vector<F, N>>, weight: &Array<Vector<F, N>>, bias: &Array<F>, output: &mut Array<F>,
    k: u32, n: u32, alpha: f32, beta: f32, #[comptime] weighted: bool,
) {
    let column = RUDA_POS_X * 4 + UNIT_POS_Y;
    let lane = UNIT_POS_X;
    let mut accumulator = Vector::<f32, N>::new(0.0f32);
    let mut position = lane;
    while position < k {
        if column < n {
            accumulator = fma(Vector::<f32,N>::cast_from(input[position as usize]),
                Vector::<f32,N>::cast_from(weight[(column * k + position) as usize]), accumulator);
        }
        position += 32;
    }
    let sum = f32::cast_from(plane_sum(Vector::vector_sum(accumulator)));
    if lane == 0 && column < n {
        let mut result = alpha * sum;
        if comptime!(weighted) { result += beta * f32::cast_from(bias[column as usize]); }
        output[column as usize] = F::cast_from(result);
    }
}

#[ruda(launch)]
pub(super) fn swiglu<F: Float + RudaElement>(input: &Array<F>, output: &mut Array<F>, width: u32) {
    let position = ABSOLUTE_POS as usize;
    if position < output.len() {
        let row = position / width as usize;
        let column = position % width as usize;
        let base = row * width as usize * 2 + column;
        let gate = f32::cast_from(input[base]);
        let activation = F::cast_from(gate / (1.0f32 + (-gate).exp()));
        output[position] = F::cast_from(f32::cast_from(activation) * f32::cast_from(input[base + width as usize]));
    }
}

#[ruda(launch)]
pub(super) fn trig<F: Float + RudaElement>(frequency: &Array<f32>, position: &Array<u32>, output: &mut Array<F>,
    width: u32, scale: f32) {
    let column = ABSOLUTE_POS as usize;
    if column < width as usize {
        let angle = f32::cast_from(position[0]) * frequency[column % (width as usize / 2)];
        output[column] = F::cast_from(angle.cos() * scale);
        output[column + width as usize] = F::cast_from(angle.sin() * scale);
    }
}

#[ruda(launch)]
pub(super) fn rotary_append<F: Float + RudaElement>(
    qkv: &Array<F>, trig: &Array<F>, query: &mut Array<F>, keys: &mut Array<F>, values: &mut Array<F>,
    position: &Array<u32>, hidden: u32, kv_width: u32, head_dim: u32, capacity: u32,
) {
    let index = ABSOLUTE_POS as usize;
    if index < hidden as usize + 2 * kv_width as usize {
        let mut local = index;
        let mut origin = 0usize;
        if index >= hidden as usize { local = index - hidden as usize; origin = hidden as usize; }
        let column = local % head_dim as usize;
        let head = local / head_dim as usize;
        let cache = (head * capacity as usize + position[0] as usize) * head_dim as usize + column;
        if index >= hidden as usize + kv_width as usize {
            local = index - hidden as usize - kv_width as usize;
            let cache = ((local / head_dim as usize) * capacity as usize + position[0] as usize) * head_dim as usize + column;
            values[cache] = qkv[index];
        } else {
            let mut partner = column + head_dim as usize / 2;
            if column >= head_dim as usize / 2 { partner = column - head_dim as usize / 2; }
            let mut rotated = f32::cast_from(qkv[origin + head * head_dim as usize + partner]);
            if column < head_dim as usize / 2 { rotated = -rotated; }
            let first = F::cast_from(f32::cast_from(qkv[index]) * f32::cast_from(trig[column]));
            let second = F::cast_from(rotated * f32::cast_from(trig[column + head_dim as usize]));
            let result = F::cast_from(f32::cast_from(first) + f32::cast_from(second));
            if index < hidden as usize { query[index] = result; }
            else { keys[cache] = result; }
        }
    }
}

#[ruda(launch)]
pub(super) fn attention_partial<F: Float + RudaElement>(
    query: &Array<F>, keys: &Array<F>, values: &Array<F>, position: &Array<u32>,
    output: &mut Array<f32>, statistics: &mut Array<f32>, heads: u32, kv_heads: u32, capacity: u32,
    scale: f32, #[comptime] width: usize, #[comptime] splits: usize,
) {
    let part_row = (RUDA_POS_X * 4 + UNIT_POS_Y) as usize;
    let head = part_row / splits;
    let split = part_row % splits;
    let lane = UNIT_POS_X as usize;
    let kv_head = head / (heads / kv_heads) as usize;
    let length = position[0] as usize + 1;
    let span = (length + splits - 1) / splits;
    let begin = split * span;
    let end = usize::min(begin + span, length);
    let mut q = Array::<f32>::new((width + 31) / 32);
    let mut acc = Array::<f32>::new((width + 31) / 32);
    #[unroll] for j in 0usize..((width + 31) / 32) {
        let column = lane + j * 32;
        q[j] = 0.0; acc[j] = 0.0;
        if column < width { q[j] = f32::cast_from(query[head * width + column]); }
    }
    let mut maximum = 0.0f32;
    let mut denominator = 0.0f32;
    for token in begin..end {
        let base = (kv_head * capacity as usize + token) * width;
        let mut dot = 0.0f32;
        #[unroll] for j in 0usize..((width + 31) / 32) {
            let column = lane + j * 32;
            if column < width { dot = fma(q[j], f32::cast_from(keys[base + column]), dot); }
        }
        let score = plane_sum(dot) * scale;
        let mut next = score;
        let mut old_scale = 0.0f32;
        if denominator != 0.0 { next = f32::max(maximum, score); old_scale = (maximum - next).exp(); }
        let probability = (score - next).exp();
        denominator = fma(denominator, old_scale, probability); maximum = next;
        #[unroll] for j in 0usize..((width + 31) / 32) {
            let column = lane + j * 32;
            if column < width { acc[j] = fma(probability, f32::cast_from(values[base + column]), acc[j] * old_scale); }
        }
    }
    #[unroll] for j in 0usize..((width + 31) / 32) {
        let column = lane + j * 32;
        if column < width { output[part_row * width + column] = acc[j]; }
    }
    if lane == 0 { statistics[part_row * 2] = maximum; statistics[part_row * 2 + 1] = denominator; }
}

#[ruda(launch)]
pub(super) fn attention_merge<F: Float + RudaElement>(
    values: &Array<f32>, statistics: &Array<f32>, output: &mut Array<F>,
    #[comptime] width: usize, #[comptime] splits: usize,
) {
    let head = RUDA_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let warp = UNIT_POS_Y as usize;
    let unit = warp * 32 + lane;
    let mut partial = SharedMemory::<f32>::new(4usize);
    let mut factors = SharedMemory::<f32>::new(splits);
    let mut maximum = f32::cast_from(f32::NEG_INFINITY);
    let mut split = unit;
    while split < splits {
        let index = (head * splits + split) * 2;
        if statistics[index + 1] != 0.0 { maximum = f32::max(maximum, statistics[index]); }
        split += 128;
    }
    let maximum = plane_max(maximum);
    if lane == 0 { partial[warp] = maximum; }
    sync_ruda();
    let mut maximum = f32::cast_from(f32::NEG_INFINITY);
    #[unroll] for i in 0usize..4 { maximum = f32::max(maximum,partial[i]); }
    sync_ruda();
    let mut denominator = 0.0f32;
    let mut split = unit;
    while split < splits {
        let index = (head * splits + split) * 2;
        let mut factor = 0.0f32;
        if statistics[index + 1] != 0.0 {
            factor = (statistics[index] - maximum).exp();
            denominator += statistics[index + 1] * factor;
        }
        factors[split] = factor;
        split += 128;
    }
    let denominator = plane_sum(denominator);
    if lane == 0 { partial[warp] = denominator; }
    sync_ruda();
    let mut denominator = 0.0f32;
    #[unroll] for i in 0usize..4 { denominator += partial[i]; }
    let mut column = unit;
    while column < width {
        let mut result = 0.0f32;
        for split in 0usize..splits { result += values[(head * splits + split) * width + column] * factors[split]; }
        if denominator != 0.0 { result /= denominator; }
        output[head * width + column] = F::cast_from(result);
        column += 128;
    }
}

#[ruda(launch)]
pub(super) fn residual<F: Float + RudaElement>(input: &Array<F>, addend: &Array<F>, output: &mut Array<F>) {
    let index = ABSOLUTE_POS as usize;
    if index < output.len() { output[index] = F::cast_from(f32::cast_from(input[index]) + f32::cast_from(addend[index])); }
}

#[ruda(launch)]
pub(super) fn advance(position: &mut Array<u32>) { if UNIT_POS == 0 { position[0] += 1u32; } }
