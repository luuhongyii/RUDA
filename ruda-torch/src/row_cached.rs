use ruda_kernel::dsl::prelude::*;

#[ruda]
fn maximum_key(value: f32) -> i32 {
    let bits = i32::reinterpret(value);
    let mut key = bits ^ ((bits >> 31) & 0x7fffffffi32);
    if value != value { key = i32::MAX; }
    key
}

#[ruda]
fn key_maximum(key: i32) -> f32 {
    f32::reinterpret(key ^ ((key >> 31) & 0x7fffffffi32))
}

#[ruda]
pub(super) fn block_reduce(value: f32, scratch: &mut SharedMemory<f32>,
    #[comptime] threads: usize, #[comptime] maximum: bool, #[comptime] slot: usize,
) -> f32 {
    let thread = UNIT_POS as usize;
    let mut reduced = 0.0f32;
    if comptime!(maximum) {
        reduced = key_maximum(plane_max(maximum_key(value)));
    } else { reduced = plane_sum(value); }
    if comptime!(threads > 32) {
    if thread % 32 == 0 { scratch[slot * (threads / 32) + thread / 32] = reduced; }
    sync_ruda();
    let mut partial = 0.0f32;
    if comptime!(maximum) { partial = f32::new(f32::NEG_INFINITY); }
    if thread % 32 < threads / 32 { partial = scratch[slot * (threads / 32) + thread % 32]; }
    if comptime!(maximum) {
        reduced = key_maximum(plane_max(maximum_key(partial)));
    } else { reduced = plane_sum(partial); }
    }
    reduced
}

#[ruda(launch)]
pub fn norm<F: Float + RudaElement, W: Float + RudaElement, B: Float + RudaElement, S: Float + RudaElement, V: Size>(
    input: &Tensor<Vector<F, V>>, weight: &Tensor<W>, bias: &Tensor<B>, out: &mut Tensor<Vector<F, V>>,
    mean: &mut Tensor<S>, rstd: &mut Tensor<S>, epsilon: f32,
    #[comptime] weighted: bool, #[comptime] biased: bool, #[comptime] layer: bool,
    #[comptime] statistics: bool, #[comptime] hf_rounding: bool,
    #[comptime] width: usize, #[comptime] threads: usize,
    #[define(V)] _vector_size: usize,
) {
    let mut row = RUDA_POS_X as usize;
    let thread = UNIT_POS as usize % threads;
    if comptime!(threads == 32) { row = row * 4usize + UNIT_POS as usize / 32usize; }
    let vector = input.vector_size().comptime();
    if row < out.len() / (width / vector) {
    let chunks = comptime!(width.div_ceil(threads * vector));
    let mut values = Array::<Vector<f32, V>>::new(chunks);
    let mut scratch = SharedMemory::<f32>::new(2 * (threads / 32));
    let mut local = 0.0f32;
    #[unroll]
    for j in 0usize..chunks {
        let col = thread + j * threads;
        let mut value = Vector::<f32, V>::new(0.0f32);
        if col < width / vector { value = Vector::cast_from(input[row * (width / vector) + col]); }
        values[j] = value;
        #[unroll] for lane in 0usize..vector {
            if comptime!(layer) { local += value[lane]; } else { local += value[lane] * value[lane]; }
        }
    }
    let total = block_reduce(local, &mut scratch, threads, false, 0usize);
    let mut mu = 0.0f32;
    let mut variance = total / width as f32;
    if comptime!(layer) {
        mu = variance;
        local = 0.0f32;
        #[unroll]
        for j in 0usize..chunks {
            let col = thread + j * threads;
            if col < width / vector {
                let centered = values[j] - Vector::<f32, V>::new(mu);
                values[j] = centered;
                #[unroll] for lane in 0usize..vector { local += centered[lane] * centered[lane]; }
            }
        }
        variance = block_reduce(local, &mut scratch, threads, false, 1usize) / width as f32;
    }
    let inv = (variance + epsilon).inverse_sqrt();
    if comptime!(statistics) {
        if thread == 0 {
            if comptime!(layer) { mean[row] = S::cast_from(mu); }
            rstd[row] = S::cast_from(inv);
        }
    }
    #[unroll]
    for j in 0usize..chunks {
        let col = thread + j * threads;
        if col < width / vector {
            let cached = values[j];
            let mut value = Vector::<F, V>::empty();
            let mut weights = Vector::<f32,V>::new(1.0f32);
            let mut biases = Vector::<f32,V>::new(0.0f32);
            if comptime!(weighted) {
                weights = Vector::cast_from(weight.slice(0usize,weight.len()).into_vectorized().with_vector_size::<V>()[col]);
            }
            if comptime!(biased) {
                biases = Vector::cast_from(bias.slice(0usize,bias.len()).into_vectorized().with_vector_size::<V>()[col]);
            }
            #[unroll] for lane in 0usize..vector {
                let mut x = cached[lane] * inv;
                if comptime!(hf_rounding) { x = f32::cast_from(F::cast_from(x)); }
                if comptime!(weighted) { x *= weights[lane]; }
                if comptime!(biased) { x += biases[lane]; }
                value[lane] = F::cast_from(x);
            }
            out[row * (width / vector) + col] = value;
        }
    }
    }
}

#[ruda(launch)]
pub fn softmax<F: Float + RudaElement, O: Float + RudaElement, V: Size>(
    input: &Tensor<Vector<F, V>>, out: &mut Tensor<Vector<O, V>>, #[comptime] logarithmic: bool,
    #[comptime] width: usize, #[comptime] threads: usize,
    #[define(V)] _vector_size: usize,
) {
    let mut row = RUDA_POS_X as usize;
    let thread = UNIT_POS as usize % threads;
    if comptime!(threads == 32) { row = row * 4usize + UNIT_POS as usize / 32usize; }
    let vector = input.vector_size().comptime();
    if row < out.len() / (width / vector) {
    let chunks = comptime!(width.div_ceil(threads * vector));
    let mut values = Array::<Vector<f32, V>>::new(chunks);
    let mut scratch = SharedMemory::<f32>::new(2 * (threads / 32));
    let mut local = f32::new(f32::NEG_INFINITY);
    #[unroll]
    for j in 0usize..chunks {
        let col = thread + j * threads;
        let mut value = Vector::<f32, V>::new(f32::new(f32::NEG_INFINITY));
        if col < width / vector { value = Vector::cast_from(input[row * (width / vector) + col]); }
        values[j] = value;
        #[unroll] for lane in 0usize..vector {
            if value[lane] > local || value[lane] != value[lane] { local = value[lane]; }
        }
    }
    let maximum = block_reduce(local, &mut scratch, threads, true, 0usize);
    local = 0.0f32;
    #[unroll]
    for j in 0usize..chunks {
        let col = thread + j * threads;
        if col < width / vector {
            let cached = values[j];
            let mut next = Vector::<f32, V>::empty();
            #[unroll] for lane in 0usize..vector {
                let shifted = cached[lane] - maximum;
                let exponential = crate::softmax::exp_fast(shifted);
                local += exponential;
                if comptime!(logarithmic) { next[lane] = shifted; } else { next[lane] = exponential; }
            }
            values[j] = next;
        }
    }
    let total = block_reduce(local, &mut scratch, threads, false, 1usize);
    let mut log_total = 0.0f32;
    if comptime!(logarithmic) { log_total = total.ln(); }
    #[unroll]
    for j in 0usize..chunks {
        let col = thread + j * threads;
        if col < width / vector {
            let cached = values[j];
            let mut value = Vector::<O, V>::empty();
            #[unroll] for lane in 0usize..vector {
                let mut x = cached[lane];
                if comptime!(logarithmic) { x -= log_total; } else { x /= total; }
                value[lane] = O::cast_from(x);
            }
            out[row * (width / vector) + col] = value;
        }
    }
    }
}

#[ruda(launch)]
pub fn softmax_backward<F: Float + RudaElement, O: Float + RudaElement, V: Size>(
    grad: &Tensor<Vector<F, V>>, value: &Tensor<Vector<F, V>>, out: &mut Tensor<Vector<O, V>>,
    #[comptime] logarithmic: bool, #[comptime] width: usize, #[comptime] threads: usize,
    #[define(V)] _vector_size: usize,
) {
    let mut row = RUDA_POS_X as usize;
    let thread = UNIT_POS as usize % threads;
    if comptime!(threads == 32) { row = row * 4usize + UNIT_POS as usize / 32usize; }
    let vector = grad.vector_size().comptime();
    if row < out.len() / (width / vector) {
        let chunks = comptime!(width.div_ceil(threads * vector));
        let mut gradients = Array::<Vector<f32, V>>::new(chunks);
        let mut values = Array::<Vector<f32, V>>::new(chunks);
        let mut scratch = SharedMemory::<f32>::new(threads / 32);
        let mut local = 0.0f32;
        #[unroll] for j in 0usize..chunks {
            let col = thread + j * threads;
            let mut g = Vector::<f32, V>::new(0.0f32);
            let mut y = Vector::<f32, V>::new(0.0f32);
            if col < width / vector {
                g = Vector::cast_from(grad[row * (width / vector) + col]);
                y = Vector::cast_from(value[row * (width / vector) + col]);
                #[unroll] for lane in 0usize..vector {
                    if comptime!(logarithmic) { local += g[lane]; }
                    else { local += g[lane] * y[lane]; }
                }
            }
            gradients[j] = g;
            values[j] = y;
        }
        let total = block_reduce(local, &mut scratch, threads, false, 0usize);
        #[unroll] for j in 0usize..chunks {
            let col = thread + j * threads;
            if col < width / vector {
                let g = gradients[j];
                let y = values[j];
                let mut result = Vector::<O, V>::empty();
                #[unroll] for lane in 0usize..vector {
                    let mut dx = 0.0f32;
                    if comptime!(logarithmic) { dx = g[lane] - crate::softmax::exp_fast(y[lane]) * total; }
                    else { dx = y[lane] * (g[lane] - total); }
                    result[lane] = O::cast_from(dx);
                }
                out[row * (width / vector) + col] = result;
            }
        }
    }
}

#[ruda(launch)]
pub fn norm_backward<F: Float + RudaElement, W: Float + RudaElement, V: Size>(
    input: &Tensor<Vector<F, V>>, weight: &Tensor<W>, grad: &Tensor<Vector<F, V>>,
    mean: &Tensor<f32>, rstd: &Tensor<f32>, out: &mut Tensor<Vector<F, V>>,
    #[comptime] weighted: bool, #[comptime] layer: bool, #[comptime] hf_rounding: bool,
    #[comptime] width: usize, #[comptime] threads: usize, #[define(V)] _vector_size: usize,
) {
    let mut row = RUDA_POS_X as usize;
    let thread = UNIT_POS as usize % threads;
    if comptime!(threads == 32) { row = row * 4usize + UNIT_POS as usize / 32usize; }
    let vector = input.vector_size().comptime();
    if row < out.len() / (width / vector) {
    let chunks = comptime!(width.div_ceil(threads * vector));
    let inv = rstd[row];
    let mut mu = 0.0f32;
    if comptime!(layer) { mu = mean[row]; }
    let mut values = Array::<Vector<f32, V>>::new(chunks);
    let mut gradients = Array::<Vector<f32, V>>::new(chunks);
    let mut scratch = SharedMemory::<f32>::new(2 * (threads / 32));
    let mut sum_g = 0.0f32;
    let mut sum_gx = 0.0f32;
    #[unroll] for j in 0usize..chunks {
        let col = thread + j * threads;
        let mut x = Vector::<f32, V>::new(0.0f32);
        let mut g = Vector::<f32, V>::new(0.0f32);
        if col < width / vector {
            x = Vector::cast_from(input[row * (width / vector) + col]);
            g = Vector::cast_from(grad[row * (width / vector) + col]);
            let mut weights = Vector::<f32,V>::new(1.0f32);
            if comptime!(weighted) {
                weights = Vector::cast_from(weight.slice(0usize,weight.len()).into_vectorized().with_vector_size::<V>()[col]);
            }
            #[unroll] for lane in 0usize..vector {
                let mut v = x[lane];
                let mut dy = g[lane];
                if comptime!(weighted) { dy *= weights[lane]; }
                if comptime!(hf_rounding) { dy = f32::cast_from(F::cast_from(dy)); }
                if comptime!(layer) {
                    v = (v - mu) * inv;
                    sum_g += dy;
                    sum_gx += dy * v;
                } else { sum_gx += dy * v * inv; }
                x[lane] = v;
                g[lane] = dy;
            }
        }
        values[j] = x;
        gradients[j] = g;
    }
    let correction = block_reduce(sum_gx, &mut scratch, threads, false, 0usize) / width as f32;
    let mut mean_grad = 0.0f32;
    if comptime!(layer) { mean_grad = block_reduce(sum_g, &mut scratch, threads, false, 1usize) / width as f32; }
    #[unroll] for j in 0usize..chunks {
        let col = thread + j * threads;
        if col < width / vector {
            let x = values[j];
            let g = gradients[j];
            let mut result = Vector::<F, V>::empty();
            #[unroll] for lane in 0usize..vector {
                let mut dx = 0.0f32;
                if comptime!(layer) { dx = inv * (g[lane] - mean_grad - x[lane] * correction); }
                else { dx = inv * (g[lane] - x[lane] * inv * correction); }
                result[lane] = F::cast_from(dx);
            }
            out[row * (width / vector) + col] = result;
        }
    }
    }
}

pub(super) fn threads(width: usize) -> usize { if width <= 4096 { 128 } else { 256 } }

pub(super) fn norm_threads(width: usize, rows: usize) -> usize {
    if width <= 256 || width <= 1024 && rows >= 4 {32} else {threads(width)}
}

pub(super) fn norm_grid(rows: usize, threads: usize) -> RudaCount {
    RudaCount::Static(u32::try_from(if threads == 32 {rows.div_ceil(4)} else {rows})
        .expect("normalization grid overflow"),1,1)
}

pub(super) fn norm_block(threads: usize) -> RudaDim { RudaDim::new_1d(threads.max(128) as u32) }

pub(super) fn eligible(width: usize) -> bool { (1..=16384).contains(&width) }

pub(super) fn vector_size(width: usize, input: &super::View, output: &super::View) -> usize {
    if width % 4 == 0 && [input, output].iter().all(|v|
        v.handle.offset_start.unwrap_or(0) % (4 * super::element_bytes(v.dtype)) as u64 == 0) { 4 } else { 1 }
}

pub(super) fn memory_vector_size(client: &super::ComputeClient<super::CudaRuntime>, width: usize,
    views: &[&super::View], maximum: usize) -> usize {
    let bytes = (client.properties().hardware.load_width as usize / 8).min(32);
    [16usize, 8, 4, 2, 1].into_iter().find(|&vector| vector <= maximum && width % vector == 0
        && views.iter().all(|view| {
            let size = vector * super::element_bytes(view.dtype);
            size <= bytes && view.handle.offset_start.unwrap_or(0) % size as u64 == 0
        })).unwrap_or(1)
}

pub(super) fn packed(view: &super::View) -> bool {
    let mut stride = 1usize;
    for (&dim, &actual) in view.shape.iter().zip(&view.strides).rev() {
        if dim > 1 && actual != stride { return false; }
        stride = stride.checked_mul(dim).expect("row stride overflow");
    }
    true
}
