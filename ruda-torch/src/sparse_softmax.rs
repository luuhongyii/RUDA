use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn position() -> usize {
    (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize
}

#[ruda]
fn compare(a: &Tensor<i64>, ai: usize, b: &Tensor<i64>, bi: usize) -> i32 {
    let mut axis = 0usize; let mut result = 0i32;
    while axis < a.shape(0) && result == 0i32 {
        let x = a[axis * a.stride(0) + ai * a.stride(1)];
        let y = b[axis * b.stride(0) + bi * b.stride(1)];
        if x < y { result = -1i32; } else if x > y { result = 1i32; }
        axis += 1usize;
    }
    result
}

#[ruda(launch)]
fn match_rows(output_indices: &Tensor<i64>, grad_indices: &Tensor<i64>, matched: &mut Tensor<u32>) {
    let row = position();
    if row < matched.len() {
        let mut low = 0usize; let mut high = grad_indices.shape(1);
        while low < high {
            let mid = low + (high - low) / 2usize;
            if compare(output_indices, row, grad_indices, mid) > 0i32 { low = mid + 1usize; }
            else { high = mid; }
        }
        let mut found = 4294967295u32;
        if low < grad_indices.shape(1) {
            if compare(output_indices, row, grad_indices, low) == 0i32 { found = u32::cast_from(low); }
        }
        matched[row] = found;
    }
}

#[ruda(launch)]
fn forward<S: Float + RudaElement>(
    values: &Tensor<S>, order: &Tensor<u32>, starts: &Tensor<u32>, count: &Tensor<i64>, output: &mut Tensor<S>,
    #[comptime] logarithmic: bool,
) {
    let p = position(); let width = values.shape(1);
    if p < values.len() {
        let pool = p / width; let column = p % width; let pools = usize::cast_from(count[0]);
        if pool < pools {
            let begin = starts[pool] as usize; let mut end = order.len();
            if pool + 1usize < pools { end = starts[pool + 1usize] as usize; }
            let mut maximum = S::new(f32::NEG_INFINITY);
            for entry in begin..end {
                let offset = order[entry] as usize * width + column;
                let value = values[address(values, offset)];
                if value > maximum || value != value { maximum = value; }
            }
            let mut total = S::new(0.0);
            for entry in begin..end {
                let offset = order[entry] as usize * width + column;
                let value = (values[address(values, offset)] - maximum).exp();
                if comptime!(!logarithmic) { output[address(output, offset)] = value; }
                total += value;
            }
            if comptime!(logarithmic) {
                let log_total = total.ln();
                for entry in begin..end {
                    let offset = order[entry] as usize * width + column;
                    output[address(output, offset)] = values[address(values, offset)] - maximum - log_total;
                }
            } else {
                let inverse = 1.0f64 / f64::cast_from(total);
                for entry in begin..end {
                    let offset = address(output, order[entry] as usize * width + column);
                    output[offset] = S::cast_from(f64::cast_from(output[offset]) * inverse);
                }
            }
        }
    }
}

#[ruda(launch)]
fn backward<S: Float + RudaElement>(
    output: &Tensor<S>, grad: &Tensor<S>, matched: &Tensor<u32>, order: &Tensor<u32>,
    starts: &Tensor<u32>, count: &Tensor<i64>, result: &mut Tensor<S>, #[comptime] logarithmic: bool,
) {
    let p = position(); let width = output.shape(1);
    if p < output.len() {
        let pool = p / width; let column = p % width; let pools = usize::cast_from(count[0]);
        if pool < pools {
            let begin = starts[pool] as usize; let mut end = order.len();
            if pool + 1usize < pools { end = starts[pool + 1usize] as usize; }
            let mut total = S::new(0.0);
            for entry in begin..end {
                let row = order[entry] as usize; let source = matched[row];
                if source != 4294967295u32 {
                    let gradient = grad[address(grad, source as usize * width + column)];
                    if comptime!(logarithmic) { total -= gradient; }
                    else { total -= output[address(output, row * width + column)] * gradient; }
                }
            }
            for entry in begin..end {
                let row = order[entry] as usize; let source = matched[row]; let offset = row * width + column;
                let value = output[address(output, offset)]; let mut result_value = S::new(0.0);
                if source != 4294967295u32 {
                    let gradient = grad[address(grad, source as usize * width + column)];
                    if comptime!(logarithmic) { result_value = gradient + value.exp() * total; }
                    else { result_value = value * (gradient + total); }
                } else {
                    if comptime!(logarithmic) { result_value = value.exp() * total; }
                    else { result_value = value * total; }
                }
                result[address(result, offset)] = result_value;
            }
        }
    }
}

#[ruda(launch)]
fn align<S: Float + RudaElement>(
    grad: &Tensor<S>, matched: &Tensor<u32>, output: &mut Tensor<S>, present: &mut Tensor<u8>,
) {
    let p = position();
    if p < output.len() {
        let width = output.shape(1); let row = p / width; let column = p % width;
        let source = matched[row]; let found = source != 4294967295u32; let mut value = S::new(0.0);
        if found { value = grad[address(grad, source as usize * width + column)]; }
        output[address(output, p)] = value;
        if column == 0usize { present[row] = u8::cast_from(found); }
    }
}

fn grid(items: usize) -> RudaCount {
    assert!(items > 0 && items <= u32::MAX as usize);
    let blocks = items.div_ceil(256); let x = blocks.min(131072);
    RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1)
}

fn packed(value: &View, length: usize, dtype: u32) {
    assert_eq!(value.dtype, dtype); assert_eq!(value.shape, [length]); assert_eq!(value.strides, [1]);
}

fn matrix(value: &View, dtype: u32) {
    assert_eq!(value.dtype, dtype); assert_eq!(value.shape.len(), 2);
}

fn pools(order: &View, starts: &View, count: &View, rows: usize) {
    packed(order, rows, 5); packed(starts, rows, 5); packed(count, 1, 4);
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64]) {
    assert!((530..=532).contains(&op));
    if op == 532 { assert!(ints.is_empty()); }
    else { assert_eq!(ints.len(), 1); assert!(matches!(ints[0], 0 | 1)); }
    assert_eq!(xs.len(), if op == 531 { 8 } else { 5 });
    let result = &xs[if op == 530 { 4 } else if op == 531 { 7 } else { 3 }];
    let dtype = result.dtype; assert!(matches!(dtype, 0 | 9)); matrix(result, dtype);
    let rows = result.shape[0]; let width = result.shape[1];
    if op == 530 {
        matrix(&xs[0], dtype); assert_eq!(xs[0].shape, result.shape); pools(&xs[1], &xs[2], &xs[3], rows);
    } else {
        for indices in &xs[..2] { matrix(indices, 4); }
        assert_eq!(xs[0].shape[0], xs[1].shape[0]); assert_eq!(xs[0].shape[1], rows);
        let grad = &xs[if op == 531 { 3 } else { 2 }]; matrix(grad, dtype);
        assert_eq!(grad.shape, [xs[1].shape[1], width]);
        if op == 531 {
            matrix(&xs[2], dtype); assert_eq!(xs[2].shape, result.shape); pools(&xs[4], &xs[5], &xs[6], rows);
        } else { packed(&xs[4], rows, 3); }
    }
    if result.len == 0 { return; }
    let c = client(); let block = RudaDim::new_1d(256);
    if op == 530 {
        macro_rules! run { ($s:ty) => { unsafe {
            forward::launch::<$s, CudaRuntime>(&c, grid(result.len), block, xs[0].arg(), xs[1].arg(),
                xs[2].arg(), xs[3].arg(), result.arg(), ints[0] != 0);
        }}; }
        match dtype { 0 => run!(f32), 9 => run!(f64), _ => unreachable!() }
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
    } else {
        let matched = View::packed(c.empty(rows.checked_mul(4).expect("sparse softmax gradient map overflow")), vec![rows], 5);
        scientific::indexable(&matched);
        unsafe { match_rows::launch::<CudaRuntime>(&c, grid(rows), block, xs[0].arg(), xs[1].arg(), matched.arg()); }
        if op == 531 {
            macro_rules! run { ($s:ty) => { unsafe {
                backward::launch::<$s, CudaRuntime>(&c, grid(result.len), block, xs[2].arg(), xs[3].arg(), matched.arg(),
                    xs[4].arg(), xs[5].arg(), xs[6].arg(), result.arg(), ints[0] != 0);
            }}; }
            match dtype { 0 => run!(f32), 9 => run!(f64), _ => unreachable!() }
        } else {
            macro_rules! run { ($s:ty) => { unsafe {
                align::launch::<$s, CudaRuntime>(&c, grid(result.len), block, xs[2].arg(), matched.arg(), result.arg(), xs[4].arg());
            }}; }
            match dtype { 0 => run!(f32), 9 => run!(f64), _ => unreachable!() }
        }
        LAUNCHES.fetch_add(2, Ordering::Relaxed);
    }
    finish_dispatch(&c);
}
