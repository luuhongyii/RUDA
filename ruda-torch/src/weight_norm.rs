use super::*;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn first_dim<S: Float + RudaElement, F: Float + RudaElement>(
    v: &Tensor<S>, g: &Tensor<S>, grad: &Tensor<S>, norms: &mut Tensor<F>,
    output: &mut Tensor<S>, grad_g: &mut Tensor<S>, width: usize, #[comptime] backward: bool,
) {
    let row = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let start = row * width;
    let mut sum = F::new(0.0);
    let mut column = lane;
    while column < width {
        let x = F::cast_from(v[start + column]);
        let mut other = x;
        if comptime!(backward) { other = F::cast_from(grad[start + column]); }
        sum += other * x;
        column += 256usize;
    }
    let mut shared = SharedMemory::<F>::new(256usize);
    shared[lane] = sum;
    sync_ruda();
    let mut offset = 128usize;
    while offset > 0 {
        if lane < offset { shared[lane] = shared[lane] + shared[lane + offset]; }
        sync_ruda();
        offset /= 2usize;
    }
    let result = shared[0];
    let mut norm = F::new(0.0);
    if comptime!(backward) { norm = norms[row]; }
    else {
        norm = F::cast_from(f32::cast_from(result).sqrt());
        if lane == 0 { norms[row] = norm; }
    }
    let inverse = F::new(1.0) / norm;
    let inverse_cube = inverse * inverse * inverse;
    if comptime!(backward) {
        if lane == 0 { grad_g[row] = S::cast_from(result * inverse); }
    }
    let magnitude = F::cast_from(g[row]);
    column = lane;
    while column < width {
        let position = start + column;
        let x = F::cast_from(v[position]);
        let mut value = magnitude * x * inverse;
        if comptime!(backward) {
            value = magnitude * (inverse * F::cast_from(grad[position]) - inverse_cube * x * result);
        }
        output[position] = S::cast_from(value);
        column += 256usize;
    }
}

#[ruda(launch)]
fn last_dim<S: Float + RudaElement, F: Float + RudaElement>(
    v: &Tensor<S>, g: &Tensor<S>, grad: &Tensor<S>, norms: &mut Tensor<F>,
    output: &mut Tensor<S>, grad_g: &mut Tensor<S>, rows: usize, columns: usize,
    #[comptime] backward: bool,
) {
    let lane = UNIT_POS as usize;
    let x_lane = lane % 16usize;
    let y_lane = lane / 16usize;
    let column = RUDA_POS_X as usize * 16usize + x_lane;
    let mut sum = F::new(0.0);
    let mut row = y_lane;
    if column < columns {
        while row < rows {
            let position = row * columns + column;
            let x = F::cast_from(v[position]);
            let mut other = x;
            if comptime!(backward) { other = F::cast_from(grad[position]); }
            sum += other * x;
            row += 64usize;
        }
    }
    let mut shared = SharedMemory::<F>::new(1024usize);
    shared[lane] = sum;
    sync_ruda();
    let mut offset = 512usize;
    while offset >= 16usize {
        if lane < offset { shared[lane] = shared[lane] + shared[lane + offset]; }
        sync_ruda();
        offset /= 2usize;
    }
    let result = shared[x_lane];
    let mut reciprocal = SharedMemory::<F>::new(16usize);
    if comptime!(!backward) {
        if y_lane == 0 && column < columns {
            let norm = F::cast_from(f32::cast_from(result).sqrt());
            norms[column] = norm;
            reciprocal[x_lane] = F::new(1.0) / norm;
        }
    }
    sync_ruda();
    if column < columns {
        let mut inverse = F::new(0.0);
        if comptime!(backward) { inverse = F::new(1.0) / norms[column]; }
        else { inverse = reciprocal[x_lane]; }
        let inverse_cube = inverse * inverse * inverse;
        if comptime!(backward) {
            if y_lane == 0 { grad_g[column] = S::cast_from(result * inverse); }
        }
        let magnitude = F::cast_from(g[column]);
        row = y_lane;
        while row < rows {
            let position = row * columns + column;
            let x = F::cast_from(v[position]);
            let mut value = magnitude * x * inverse;
            if comptime!(backward) {
                value = magnitude * (inverse * F::cast_from(grad[position]) - inverse_cube * x * result);
            }
            output[position] = S::cast_from(value);
            row += 64usize;
        }
    }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64]) {
    assert!(matches!(op, 447 | 448));
    let backward = op == 448;
    assert_eq!(xs.len(), if backward { 6 } else { 4 });
    assert_eq!(ints.len(), 3);
    assert!(ints[0] >= 0 && ints[1] >= 0 && matches!(ints[2], 0 | 1));
    let rows = usize::try_from(ints[0]).expect("weight norm row count overflow");
    let columns = usize::try_from(ints[1]).expect("weight norm column count overflow");
    let first = ints[2] != 0;
    let groups = if first { rows } else { columns };
    let (v, g, grad, norms, output, grad_g) = if backward {
        (&xs[1], &xs[2], &xs[0], &xs[3], &xs[4], &xs[5])
    } else { (&xs[0], &xs[1], &xs[0], &xs[3], &xs[2], &xs[2]) };
    assert!(matches!(v.dtype, 0 | 1 | 2 | 9));
    assert_eq!(norms.dtype, if v.dtype == 9 { 9 } else { 0 });
    assert_eq!(v.len, rows.checked_mul(columns).expect("weight norm size overflow"));
    for value in [g, grad, output, grad_g] { assert_eq!(value.dtype, v.dtype); }
    assert_eq!(grad.len, v.len);
    assert_eq!(output.len, v.len);
    assert!(g.len >= groups);
    assert_eq!(norms.len, g.len);
    if backward { assert_eq!(grad_g.len, g.len); }
    for value in xs {
        assert_eq!(value.shape, vec![value.len]);
        assert_eq!(value.strides, vec![1usize]);
    }
    if groups == 0 { return; }
    let c = client();
    let blocks = if first { groups } else { groups.div_ceil(16) };
    let grid = RudaCount::Static(u32::try_from(blocks).expect("weight norm launch overflow"), 1, 1);
    macro_rules! run { ($s:ty, $f:ty) => { unsafe {
        if first {
            first_dim::launch::<$s, $f, CudaRuntime>(&c, grid, RudaDim::new_1d(256),
                v.arg(), g.arg(), grad.arg(), norms.arg(), output.arg(), grad_g.arg(), columns, backward);
        } else {
            last_dim::launch::<$s, $f, CudaRuntime>(&c, grid, RudaDim::new_2d(16, 64),
                v.arg(), g.arg(), grad.arg(), norms.arg(), output.arg(), grad_g.arg(), rows, columns, backward);
        }
    }}; }
    match v.dtype {
        0 => run!(f32, f32), 1 => run!(f16, f32), 2 => run!(bf16, f32),
        9 => run!(f64, f64), _ => unreachable!(),
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    finish_dispatch(&c);
}
