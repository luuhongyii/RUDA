use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn finite(value: f32) -> bool {
    value == value && value.abs() != f32::INFINITY
}

#[ruda(launch)]
fn unscale<S: Float + RudaElement, F: Float + RudaElement>(
    gradients: &mut Sequence<Tensor<S>>, found_inf: &mut Tensor<Atomic<f32>>, inv_scale: &Tensor<f32>,
) {
    let block = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let mut first = 0usize;
    #[unroll]
    for tensor in 0..gradients.len() {
        let grad = gradients.index_mut(tensor);
        let blocks = grad.len() / 256 + usize::cast_from(grad.len() % 256 != 0);
        if block >= first && block < first + blocks {
            let p = (block - first) * 256 + lane;
            if p < grad.len() {
                let offset = address(grad, p);
                let value = F::cast_from(grad[offset]);
                if !finite(f32::cast_from(value)) { found_inf[0].store(1.0f32); }
                let inverse = inv_scale[0];
                let mut result = value;
                if inverse != 1.0f32 { result = value * F::cast_from(inverse); }
                grad[offset] = S::cast_from(result);
            }
        }
        first += blocks;
    }
}

#[ruda(launch)]
fn update_scale(
    scale: &mut Tensor<f32>, tracker: &mut Tensor<i32>, found_inf: &Tensor<f32>,
    growth: f64, backoff: f64, interval: i32,
) {
    if found_inf[0] != 0.0f32 {
        scale[0] = f32::cast_from(f64::cast_from(scale[0]) * backoff);
        tracker[0] = 0i32;
    } else {
        let successful = i32::cast_from(u32::cast_from(tracker[0]) + 1u32);
        if successful == interval {
            let next = f32::cast_from(f64::cast_from(scale[0]) * growth);
            if finite(next) { scale[0] = next; }
            tracker[0] = 0i32;
        } else {
            tracker[0] = successful;
        }
    }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64], values: &[f64]) {
    assert!(matches!(op, 434 | 435));
    if op == 435 {
        assert_eq!(xs.len(), 3);
        assert_eq!(ints.len(), 1);
        assert_eq!(values.len(), 2);
        assert_eq!((xs[0].dtype, xs[0].len), (0, 1));
        assert_eq!((xs[1].dtype, xs[1].len), (5, 1));
        assert_eq!((xs[2].dtype, xs[2].len), (0, 1));
        let c = client();
        unsafe {
            update_scale::launch::<CudaRuntime>(&c, RudaCount::Static(1, 1, 1), RudaDim::new_1d(1),
                xs[0].arg(), xs[1].arg(), xs[2].arg(), values[0], values[1], ints[0] as i32);
        }
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
        finish_dispatch(&c);
        return;
    }
    assert!(ints.is_empty() && values.is_empty() && xs.len() >= 2);
    let count = xs.len() - 2;
    let found_inf = &xs[count];
    let inv_scale = &xs[count + 1];
    assert_eq!((found_inf.dtype, found_inf.len), (0, 1));
    assert_eq!((inv_scale.dtype, inv_scale.len), (0, 1));
    assert!(xs[..count].iter().all(|x| matches!(x.dtype, 0 | 1 | 9)));
    if xs[..count].iter().all(|x| x.len == 0) { return; }
    let c = client();
    for dtype in [0, 1, 9] {
        let gradients: Vec<_> = xs[..count].iter().filter(|x| x.dtype == dtype && x.len != 0).collect();
        for batch in gradients.chunks(32) {
            let blocks: usize = batch.iter().map(|x| x.len.div_ceil(256)).sum();
            let grid = RudaCount::Static(u32::try_from(blocks).expect("AMP batch launch overflow"), 1, 1);
            macro_rules! run { ($s:ty, $f:ty) => { unsafe {
                let args = batch.iter().map(|x| x.arg()).collect::<SequenceArg<CudaRuntime, Tensor<$s>>>();
                unscale::launch::<$s, $f, CudaRuntime>(&c, grid, RudaDim::new_1d(256),
                    args, found_inf.arg(), inv_scale.arg());
            }}; }
            match dtype { 0 => run!(f32, f32), 1 => run!(f16, f32), 9 => run!(f64, f64), _ => unreachable!() }
            LAUNCHES.fetch_add(1, Ordering::Relaxed);
        }
    }
    finish_dispatch(&c);
}
