use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn position() -> usize {
    (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize
}

#[ruda]
fn row_for_entry<I: Numeric>(pointers: &Tensor<I>, entry: usize) -> usize {
    let mut low = 0usize; let mut high = pointers.len();
    while low < high {
        let mid = low + (high - low) / 2usize;
        if usize::cast_from(pointers[address(pointers, mid)]) <= entry { low = mid + 1usize; }
        else { high = mid; }
    }
    low - 1usize
}

#[ruda(launch)]
fn entry_rows<I: Numeric + RudaElement>(pointers: &Tensor<I>, rows: &mut Tensor<u32>) {
    let entry = position();
    if entry < rows.len() { rows[entry] = u32::cast_from(row_for_entry(pointers, entry)); }
}

#[ruda(launch)]
fn forward<I: Numeric + RudaElement, S: Float + RudaElement, A: Float>(
    pointers: &Tensor<I>, columns: &Tensor<I>, values: &Tensor<S>, other: &Tensor<S>,
    output: &mut Tensor<S>, arguments: &mut Tensor<I>, #[comptime] mode: u32, #[comptime] save_arg: bool,
) {
    let p = position();
    if p < output.len() {
        let width = output.shape(1); let row = p / width; let column = p % width;
        let begin = usize::cast_from(pointers[address(pointers, row)]);
        let end = usize::cast_from(pointers[address(pointers, row + 1usize)]);
        let mut total = A::new(0.0); let mut winner = values.len();
        if begin != end {
            if comptime!(mode == 2) { total = A::new(f32::NEG_INFINITY); }
            else if comptime!(mode == 3) { total = A::new(f32::INFINITY); }
            for entry in begin..end {
                let source = usize::cast_from(columns[address(columns, entry)]);
                let product = A::cast_from(values[address(values, entry)]) * A::cast_from(other[address(other, source * width + column)]);
                if comptime!(mode <= 1) { total += product; }
                else if (mode == 2u32 && product > total) || (mode == 3u32 && product < total) || product != product {
                    total = product; winner = entry;
                }
            }
        }
        let mut result = S::cast_from(total);
        if comptime!(mode == 1) {
            if begin != end { result = S::cast_from(A::cast_from(result) / A::cast_from(end - begin)); }
        }
        output[address(output, p)] = result;
        if comptime!(save_arg) { arguments[address(arguments, p)] = I::cast_from(winner); }
    }
}

#[ruda(launch)]
fn backward_values<I: Numeric + RudaElement, S: Float + RudaElement, A: Float>(
    pointers: &Tensor<I>, columns: &Tensor<I>, other: &Tensor<S>, grad: &Tensor<S>,
    arguments: &Tensor<I>, output: &mut Tensor<S>, #[comptime] mode: u32,
) {
    let entry = position();
    if entry < output.len() {
        let row = row_for_entry(pointers, entry); let source = usize::cast_from(columns[address(columns, entry)]);
        let width = grad.shape(1); let mut total = A::new(0.0);
        for column in 0..width {
            let offset = row * width + column; let mut selected = true;
            if comptime!(mode >= 2) { selected = usize::cast_from(arguments[address(arguments, offset)]) == entry; }
            if selected {
                let term = A::cast_from(other[address(other, source * width + column)]) * A::cast_from(grad[address(grad, offset)]);
                if comptime!(mode >= 2) {
                    total = A::cast_from(S::cast_from(total + A::cast_from(S::cast_from(term))));
                } else { total += term; }
            }
        }
        let mut result = S::cast_from(total);
        if comptime!(mode == 1) {
            let begin = usize::cast_from(pointers[address(pointers, row)]);
            let end = usize::cast_from(pointers[address(pointers, row + 1usize)]);
            result = S::cast_from(A::cast_from(result) / A::cast_from(S::cast_from(A::cast_from(end - begin))));
        }
        output[address(output, entry)] = result;
    }
}

#[ruda(launch)]
fn backward_other<I: Numeric + RudaElement, S: Float + RudaElement, A: Float>(
    pointers: &Tensor<I>, values: &Tensor<S>, grad: &Tensor<S>, arguments: &Tensor<I>,
    column_pointers: &Tensor<I>, order: &Tensor<i64>, rows: &Tensor<u32>, output: &mut Tensor<S>, #[comptime] mode: u32,
) {
    let p = position();
    if p < output.len() {
        let width = output.shape(1); let target = p / width; let column = p % width;
        let begin = usize::cast_from(column_pointers[address(column_pointers, target)]);
        let end = usize::cast_from(column_pointers[address(column_pointers, target + 1usize)]);
        let mut total = A::new(0.0);
        for slot in begin..end {
            let entry = usize::cast_from(order[address(order, slot)]); let row = rows[entry] as usize;
            let offset = row * width + column; let mut selected = true;
            if comptime!(mode >= 2) { selected = usize::cast_from(arguments[address(arguments, offset)]) == entry; }
            if selected {
                let mut value = values[address(values, entry)];
                if comptime!(mode == 1) {
                    let row_begin = usize::cast_from(pointers[address(pointers, row)]);
                    let row_end = usize::cast_from(pointers[address(pointers, row + 1usize)]);
                    value = S::cast_from(A::cast_from(value) / A::cast_from(S::cast_from(A::cast_from(row_end - row_begin))));
                }
                let term = A::cast_from(value) * A::cast_from(grad[address(grad, offset)]);
                if comptime!(mode >= 2) {
                    total = A::cast_from(S::cast_from(total + A::cast_from(S::cast_from(term))));
                } else { total += term; }
            }
        }
        output[address(output, p)] = S::cast_from(total);
    }
}

fn grid(items: usize) -> RudaCount {
    assert!(items > 0 && items <= u32::MAX as usize);
    let blocks = items.div_ceil(256); let x = blocks.min(131072);
    RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1)
}

fn vector(v: &View, dtype: u32) { assert_eq!(v.dtype, dtype); assert_eq!(v.shape.len(), 1); }
fn matrix(v: &View, dtype: u32) { assert_eq!(v.dtype, dtype); assert_eq!(v.shape.len(), 2); }

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64]) {
    assert!((533..=535).contains(&op)); assert_eq!(ints.len(), 1); assert!((0..=3).contains(&ints[0]));
    assert_eq!(xs.len(), if op == 535 { 7 } else { 6 }); let mode = ints[0] as u32;
    let pointers = &xs[0]; let index_dtype = pointers.dtype; assert!(matches!(index_dtype, 4 | 5));
    vector(pointers, index_dtype); assert!(pointers.len > 0); let rows = pointers.len - 1;
    let output = &xs[if op == 533 { 4 } else if op == 534 { 5 } else { 6 }];
    let dtype = output.dtype; assert!(matches!(dtype, 0..=2 | 9));
    let arguments = &xs[if op == 533 { 5 } else if op == 534 { 4 } else { 3 }]; assert_eq!(arguments.dtype, index_dtype);
    if op == 533 {
        vector(&xs[1], index_dtype); vector(&xs[2], dtype); assert_eq!(xs[1].len, xs[2].len);
        matrix(&xs[3], dtype); matrix(output, dtype); assert_eq!(output.shape, [rows, xs[3].shape[1]]);
        if arguments.len > 0 { assert!(mode >= 2); assert_eq!(arguments.shape, output.shape); }
    } else if op == 534 {
        vector(&xs[1], index_dtype); matrix(&xs[2], dtype); matrix(&xs[3], dtype); vector(output, dtype);
        assert_eq!(xs[1].len, output.len); assert_eq!(xs[3].shape, [rows, xs[2].shape[1]]);
        if mode >= 2 { assert_eq!(arguments.shape, xs[3].shape); }
    } else {
        vector(&xs[1], dtype); matrix(&xs[2], dtype); vector(&xs[4], index_dtype); vector(&xs[5], 4); matrix(output, dtype);
        assert_eq!(xs[4].len, output.shape[0] + 1); assert_eq!(xs[1].len, xs[5].len);
        assert_eq!(xs[2].shape, [rows, output.shape[1]]);
        if mode >= 2 { assert_eq!(arguments.shape, xs[2].shape); }
    }
    if output.len == 0 { return; }
    let c = client(); let block = RudaDim::new_1d(256);
    let entry_map = if op == 535 {
        let count = xs[1].len; assert!(count > 0);
        let map = View::packed(c.empty(count.checked_mul(4).expect("SpMM row map overflow")), vec![count], 5);
        scientific::indexable(&map);
        macro_rules! run { ($i:ty) => { unsafe { entry_rows::launch::<$i, CudaRuntime>(&c, grid(count), block, pointers.arg(), map.arg()); }}; }
        if index_dtype == 4 { run!(i64); } else { run!(i32); }
        Some(map)
    } else { None };
    macro_rules! run { ($i:ty, $s:ty, $a:ty) => { unsafe {
        if op == 533 {
            forward::launch::<$i, $s, $a, CudaRuntime>(&c, grid(output.len), block, pointers.arg(), xs[1].arg(), xs[2].arg(),
                xs[3].arg(), output.arg(), arguments.arg(), mode, arguments.len != 0);
        } else if op == 534 {
            backward_values::launch::<$i, $s, $a, CudaRuntime>(&c, grid(output.len), block, pointers.arg(), xs[1].arg(),
                xs[2].arg(), xs[3].arg(), arguments.arg(), output.arg(), mode);
        } else {
            backward_other::launch::<$i, $s, $a, CudaRuntime>(&c, grid(output.len), block, pointers.arg(), xs[1].arg(),
                xs[2].arg(), arguments.arg(), xs[4].arg(), xs[5].arg(), entry_map.as_ref().unwrap().arg(), output.arg(), mode);
        }
    }}; }
    macro_rules! typed { ($s:ty, $a:ty) => { if index_dtype == 4 { run!(i64, $s, $a); } else { run!(i32, $s, $a); } }; }
    match dtype { 0 => typed!(f32, f32), 1 => typed!(f16, f32), 2 => typed!(bf16, f32), 9 => typed!(f64, f64), _ => unreachable!() }
    LAUNCHES.fetch_add(if op == 535 { 2 } else { 1 }, Ordering::Relaxed); finish_dispatch(&c);
}
