use super::*;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn control(controls: &Sequence<Tensor<f32>>, #[comptime] flags: u32, #[comptime] slot: u32) -> f32 {
    let index = comptime!((flags & ((1u32 << slot) - 1)).count_ones() as usize);
    controls.index(index)[0]
}

#[ruda]
fn learning_rate(controls: &Sequence<Tensor<f32>>, host_lr: f64, #[comptime] flags: u32) -> f64 {
    let mut lr = host_lr;
    if comptime!(flags & 1 != 0) { lr = f64::cast_from(control(controls, flags, 0)); }
    lr
}

#[ruda]
fn skip(controls: &Sequence<Tensor<f32>>, #[comptime] flags: u32) -> bool {
    let mut result = false;
    if comptime!(flags & 4 != 0) { result = control(controls, flags, 2) == 1.0f32; }
    result
}

#[ruda(launch)]
fn adam<P: Float + RudaElement, M: Float + RudaElement, F: Float + RudaElement>(
    params: &mut Sequence<Tensor<P>>, grads: &mut Sequence<Tensor<P>>,
    moments: &mut Sequence<Tensor<M>>, variances: &mut Sequence<Tensor<M>>,
    maxima: &mut Sequence<Tensor<M>>, steps: &Sequence<Tensor<f32>>, controls: &Sequence<Tensor<f32>>,
    host_lr: f64, beta1: f64, beta2: f64, decay: f64, epsilon: f64,
    #[comptime] flags: u32, #[comptime] maximize: bool, #[comptime] amsgrad: bool,
    #[comptime] adamw: bool, #[comptime] mixed: bool,
) {
    if skip(controls, flags) { terminate!(); }
    let lr = F::cast_from(learning_rate(controls, host_lr, flags));
    let b1 = F::cast_from(beta1);
    let b2 = F::cast_from(beta2);
    let wd = F::cast_from(decay);
    let eps = F::cast_from(epsilon);
    let block = RUDA_POS_X as usize;
    let mut first = 0usize;
    #[unroll]
    for tensor in 0..params.len() {
        let param = params.index_mut(tensor);
        let blocks = param.len() / 65536 + usize::cast_from(param.len() % 65536 != 0);
        if block >= first && block < first + blocks {
            let step = steps.index(tensor)[0];
            let mut correction1 = F::new(0.0);
            let mut correction2 = F::new(0.0);
            if comptime!(mixed) {
                correction1 = F::cast_from(1.0f64 - beta1.powf(f64::cast_from(step)));
                correction2 = F::cast_from((1.0f64 - beta2.powf(f64::cast_from(step))).sqrt());
            } else {
                correction1 = F::new(1.0) - b1.powf(F::cast_from(step));
                correction2 = (F::new(1.0) - b2.powf(F::cast_from(step))).sqrt();
            }
            let step_size = lr / correction1;
            let grad = grads.index_mut(tensor);
            let moment = moments.index_mut(tensor);
            let variance = variances.index_mut(tensor);
            let base = (block - first) * 65536;
            let remaining = param.len() - base;
            let mut offset = UNIT_POS as usize;
            while offset < remaining && offset < 65536 {
                let p = base + offset;
                let mut value = F::cast_from(param[p]);
                let mut g = F::cast_from(grad[p]);
                if comptime!(flags & 2 != 0) { g /= F::cast_from(control(controls, flags, 1)); }
                let unscaled = g;
                if comptime!(maximize) { g = -g; }
                if wd != F::new(0.0) {
                    if comptime!(adamw) { value -= lr * wd * value; }
                    else { g += value * wd; }
                }
                let mean = fma(b1, F::cast_from(moment[p]), fma(-b1, g, g));
                let square = g * g;
                let var = fma(b2, F::cast_from(variance[p]), fma(-b2, square, square));
                let mut denominator_value = var;
                if comptime!(amsgrad) {
                    let maximum = maxima.index_mut(tensor);
                    let mut largest = F::cast_from(maximum[p]);
                    if largest < var { largest = var; }
                    denominator_value = largest;
                }
                let denominator = denominator_value.sqrt() / correction2 + eps;
                value -= step_size * mean / denominator;
                param[p] = P::cast_from(value);
                if comptime!(flags & 2 != 0 && !mixed) { grad[p] = P::cast_from(unscaled); }
                moment[p] = M::cast_from(mean);
                variance[p] = M::cast_from(var);
                if comptime!(amsgrad) {
                    let maximum = maxima.index_mut(tensor);
                    maximum[p] = M::cast_from(denominator_value);
                }
                if comptime!(flags & 2 != 0 && mixed) { grad[p] = P::cast_from(unscaled); }
                offset += 512;
            }
        }
        first += blocks;
    }
}

#[ruda(launch)]
fn sgd<S: Float + RudaElement, F: Float + RudaElement>(
    params: &mut Sequence<Tensor<S>>, grads: &mut Sequence<Tensor<S>>,
    buffers: &mut Sequence<Tensor<S>>, controls: &Sequence<Tensor<f32>>,
    host_lr: f64, decay: f64, momentum: f64, dampening: f64,
    #[comptime] flags: u32, #[comptime] maximize: bool, #[comptime] with_momentum: bool,
    #[comptime] nesterov: bool, #[comptime] first_step: bool,
) {
    if skip(controls, flags) { terminate!(); }
    let lr = F::cast_from(learning_rate(controls, host_lr, flags));
    let wd = F::cast_from(decay);
    let beta = F::cast_from(momentum);
    let damp = F::cast_from(dampening);
    let block = RUDA_POS_X as usize;
    let mut first = 0usize;
    #[unroll]
    for tensor in 0..params.len() {
        let param = params.index_mut(tensor);
        let blocks = param.len() / 65536 + usize::cast_from(param.len() % 65536 != 0);
        if block >= first && block < first + blocks {
            let grad = grads.index_mut(tensor);
            let base = (block - first) * 65536;
            let remaining = param.len() - base;
            let mut offset = UNIT_POS as usize;
            while offset < remaining && offset < 65536 {
                let p = base + offset;
                let mut value = F::cast_from(param[p]);
                let mut g = F::cast_from(grad[p]);
                if comptime!(flags & 2 != 0) {
                    g /= F::cast_from(control(controls, flags, 1));
                }
                let unscaled = g;
                let mut next_buffer = F::new(0.0);
                if comptime!(maximize) { g = -g; }
                if wd != F::new(0.0) { g += wd * value; }
                if comptime!(with_momentum) {
                    let buffer = buffers.index_mut(tensor);
                    let mut next = g;
                    if comptime!(!first_step) { next = beta * F::cast_from(buffer[p]) + (F::new(1.0) - damp) * g; }
                    next_buffer = next;
                    if comptime!(nesterov) { g += beta * next; }
                    else { g = next; }
                }
                value -= lr * g;
                param[p] = S::cast_from(value);
                if comptime!(flags & 2 != 0) { grad[p] = S::cast_from(unscaled); }
                if comptime!(with_momentum) {
                    let buffer = buffers.index_mut(tensor);
                    buffer[p] = S::cast_from(next_buffer);
                }
                offset += 512;
            }
        }
        first += blocks;
    }
}

#[ruda(launch)]
fn adagrad<S: Float + RudaElement, F: Float + RudaElement>(
    params: &mut Sequence<Tensor<S>>, grads: &mut Sequence<Tensor<S>>,
    sums: &mut Sequence<Tensor<S>>, steps: &Sequence<Tensor<f32>>, controls: &Sequence<Tensor<f32>>,
    host_lr: f64, lr_decay: f64, weight_decay: f64, eps: f64,
    #[comptime] flags: u32, #[comptime] maximize: bool,
) {
    if skip(controls, flags) { terminate!(); }
    let lr = learning_rate(controls, host_lr, flags);
    let block = RUDA_POS_X as usize;
    let mut first = 0usize;
    #[unroll]
    for tensor in 0..params.len() {
        let param = params.index_mut(tensor);
        let blocks = param.len() / 65536 + usize::cast_from(param.len() % 65536 != 0);
        if block >= first && block < first + blocks {
            let step_offset = steps.index(tensor)[0] - 1.0f32;
            let corrected_lr = lr / (1.0f64 + f64::cast_from(step_offset) * lr_decay);
            let grad = grads.index_mut(tensor);
            let sum = sums.index_mut(tensor);
            let base = (block - first) * 65536;
            let remaining = param.len() - base;
            let mut offset = UNIT_POS as usize;
            while offset < remaining && offset < 65536 {
                let p = base + offset;
                let value = F::cast_from(param[p]);
                let mut g = F::cast_from(grad[p]);
                if comptime!(flags & 2 != 0) {
                    g = F::cast_from(f64::cast_from(g) / f64::cast_from(control(controls, flags, 1)));
                }
                let unscaled = g;
                if comptime!(maximize) { g = -g; }
                if weight_decay != 0.0f64 {
                    g = F::cast_from(f64::cast_from(g) + f64::cast_from(value) * weight_decay);
                }
                let state = F::cast_from(sum[p]) + g * g;
                let next = F::cast_from(f64::cast_from(value)
                    - corrected_lr * f64::cast_from(g) / (f64::cast_from(state.sqrt()) + eps));
                param[p] = S::cast_from(next);
                if comptime!(flags & 2 != 0) { grad[p] = S::cast_from(unscaled); }
                sum[p] = S::cast_from(state);
                offset += 512;
            }
        }
        first += blocks;
    }
}

fn arguments<S: Float + RudaElement>(xs: &[View], batch: &[usize], start: usize) -> SequenceArg<CudaRuntime, Tensor<S>> {
    batch.iter().map(|&i| xs[start + i].arg()).collect()
}

fn dense(view: &View) {
    if view.len == 0 { return; }
    let mut axes: Vec<_> = (0..view.shape.len()).filter(|&d| view.shape[d] > 1).collect();
    axes.sort_by_key(|&d| view.strides[d]);
    let mut expected = 1usize;
    for d in axes { assert_eq!(view.strides[d], expected); expected *= view.shape[d]; }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64], values: &[f64]) {
    assert!((436..=439).contains(&op) && ints.len() == 6 && ints[0] > 0);
    let n = usize::try_from(ints[0]).expect("optimizer list length overflow");
    assert!((0..=7).contains(&ints[1]) && ints[2..].iter().all(|&x| matches!(x, 0 | 1)));
    let flags = ints[1] as u32;
    let variant = ints[3] != 0;
    let adam_kind = op <= 437;
    let fields = if adam_kind { 4 + usize::from(variant) } else if op == 438 { 2 + usize::from(variant) } else { 3 };
    let with_steps = op != 438;
    assert_eq!(values.len(), if adam_kind { 5 } else { 4 });
    let controls_start = n.checked_mul(fields + usize::from(with_steps)).expect("optimizer descriptor overflow");
    assert_eq!(xs.len(), controls_start + flags.count_ones() as usize);
    let code = xs[0].dtype;
    assert!(matches!(code, 0 | 1 | 2 | 9));
    let state_code = if adam_kind { xs[2 * n].dtype } else { code };
    let mixed = code != state_code;
    assert!(!mixed || (adam_kind && code == 0 && state_code == 2));
    for field in 0..fields {
        for i in 0..n {
            let x = &xs[field * n + i];
            assert_eq!(x.dtype, if adam_kind && field >= 2 { state_code } else { code });
            assert_eq!(x.shape, xs[i].shape);
            assert!(x.shape.iter().enumerate().all(|(d, &size)| size <= 1 || x.strides[d] == xs[i].strides[d]));
            dense(x);
        }
    }
    for x in &xs[fields * n..] { assert!(x.dtype == 0 && x.len > 0); }
    if op == 438 { assert!(if variant { values[2] > 0.0 } else { values[2] == 0.0 }); }
    let active: Vec<usize> = (0..n).filter(|&i| xs[i].len != 0).collect();
    if active.is_empty() { return; }
    let c = client();
    for batch in active.chunks(8) {
        let blocks: usize = batch.iter().map(|&i| xs[i].len.div_ceil(65536)).sum();
        let grid = RudaCount::Static(u32::try_from(blocks).expect("optimizer grid overflow"), 1, 1);
        let block = RudaDim::new_1d(512);
        let controls = || xs[controls_start..].iter().map(|x| x.arg()).collect::<SequenceArg<CudaRuntime, Tensor<f32>>>();
        macro_rules! run { ($p:ty, $m:ty, $f:ty) => { unsafe {
            if adam_kind {
                let maxima = if variant { arguments::<$m>(xs, batch, 4 * n) } else { SequenceArg::new() };
                adam::launch::<$p, $m, $f, CudaRuntime>(&c, grid, block,
                    arguments::<$p>(xs, batch, 0), arguments::<$p>(xs, batch, n),
                    arguments::<$m>(xs, batch, 2 * n), arguments::<$m>(xs, batch, 3 * n), maxima,
                    arguments::<f32>(xs, batch, fields * n), controls(),
                    values[0], values[1], values[2], values[3], values[4], flags, ints[2] != 0, variant, op == 437, mixed);
            } else if op == 438 {
                let buffers = if variant { arguments::<$p>(xs, batch, 2 * n) } else { SequenceArg::new() };
                sgd::launch::<$p, $f, CudaRuntime>(&c, grid, block,
                    arguments::<$p>(xs, batch, 0), arguments::<$p>(xs, batch, n), buffers, controls(),
                    values[0], values[1], values[2], values[3], flags, ints[2] != 0, variant, ints[4] != 0, ints[5] != 0);
            } else {
                adagrad::launch::<$p, $f, CudaRuntime>(&c, grid, block,
                    arguments::<$p>(xs, batch, 0), arguments::<$p>(xs, batch, n), arguments::<$p>(xs, batch, 2 * n),
                    arguments::<f32>(xs, batch, fields * n), controls(), values[0], values[1], values[2], values[3], flags, ints[2] != 0);
            }
        }}; }
        match (code, state_code) {
            (0, 0) => run!(f32, f32, f32), (0, 2) => run!(f32, bf16, f32),
            (1, 1) => run!(f16, f16, f32), (2, 2) => run!(bf16, bf16, f32),
            (9, 9) => run!(f64, f64, f64), _ => unreachable!(),
        }
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
    }
    finish_dispatch(&c);
}
