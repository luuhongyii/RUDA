use super::*;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn block_id() -> usize {
    RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize
}

#[ruda]
fn lower(columns: &Tensor<i64>, begin: usize, end: usize, column: i64) -> usize {
    let mut lo = begin;
    let mut hi = end;
    while lo < hi {
        let mid = lo + (hi - lo) / 2usize;
        if columns[mid] < column { lo = mid + 1usize; } else { hi = mid; }
    }
    lo
}

#[ruda]
fn coefficient<F: Float + RudaElement>(
    pointers: &Tensor<i64>, columns: &Tensor<i64>, values: &Tensor<F>, row: usize, column: i64,
) -> F {
    let end = pointers[row + 1usize] as usize;
    let p = lower(columns, pointers[row] as usize, end, column);
    let mut value = F::new(0.0);
    if p < end {
        if columns[p] == column { value = values[p]; }
    }
    value
}

#[ruda]
fn better<F: Float>(score: F, row: u32, best: F, selected: u32) -> bool {
    let nan = score != score;
    let best_nan = best != best;
    (nan && !best_nan) || (nan == best_nan && ((nan || score == best) && row < selected || score > best))
}

#[ruda(launch)]
fn choose_pivot<F: Float + RudaElement>(
    pointers: &Tensor<i64>, columns: &Tensor<i64>, values: &Tensor<F>,
    rank: &mut Tensor<i64>, pivots: &mut Tensor<i64>, info: &mut Tensor<i32>, column: i64,
) {
    let tid = UNIT_POS as usize;
    let n = rank.len();
    let mut best = F::new(0.0);
    let mut selected = u32::cast_from(n);
    let mut row = tid;
    while row < n {
        if rank[row] < 0i64 {
            let score = coefficient(pointers, columns, values, row, column).abs();
            if better(score, u32::cast_from(row), best, selected) {
                best = score;
                selected = u32::cast_from(row);
            }
        }
        row += 256usize;
    }
    let mut scores = SharedMemory::<F>::new(256usize);
    let mut rows = SharedMemory::<u32>::new(256usize);
    scores[tid] = best;
    rows[tid] = selected;
    sync_ruda();
    let mut stride = 128usize;
    while stride > 0usize {
        if tid < stride {
            let score = scores[tid + stride];
            let row = rows[tid + stride];
            if better(score, row, scores[tid], rows[tid]) {
                scores[tid] = score;
                rows[tid] = row;
            }
        }
        sync_ruda();
        stride /= 2usize;
    }
    if tid == 0usize {
        if rows[0] == u32::cast_from(n) || scores[0] == F::new(0.0) {
            info[0] = i32::cast_from(column + 1i64);
            pivots[column as usize] = 0i64;
        } else {
            let chosen = rows[0] as usize;
            rank[chosen] = column;
            pivots[column as usize] = i64::cast_from(chosen);
        }
    }
}

#[ruda]
fn merged_pattern(
    pointers: &Tensor<i64>, columns: &Tensor<i64>, row: usize, pivot: usize, column: i64,
    active: bool, frozen: bool, output: &mut Tensor<i64>, start: usize, #[comptime] write: bool,
) -> usize {
    let begin = pointers[row] as usize;
    let end = pointers[row + 1usize] as usize;
    if frozen {
        if comptime!(write) {
            for p in begin..end { output[start + (p - begin)] = columns[p]; }
        }
        return end - begin;
    }
    let pivot_end = pointers[pivot + 1usize] as usize;
    let mut right = pivot_end;
    if active { right = lower(columns, pointers[pivot] as usize, pivot_end, column + 1i64); }
    let mut left = begin;
    let mut count = 0usize;
    while left < end || right < pivot_end {
        let mut a = 9223372036854775807i64;
        let mut b = 9223372036854775807i64;
        if left < end { a = columns[left]; }
        if right < pivot_end { b = columns[right]; }
        let mut next = a;
        if b < a { next = b; }
        if a <= b { left += 1usize; }
        if b <= a { right += 1usize; }
        if next != column {
            if comptime!(write) { output[start + count] = next; }
            count += 1usize;
        }
    }
    count
}

#[ruda(launch)]
fn count_rows<F: Float + RudaElement>(
    pointers: &Tensor<i64>, columns: &Tensor<i64>, values: &Tensor<F>,
    rank: &Tensor<i64>, pivots: &Tensor<i64>, info: &Tensor<i32>, counts: &mut Tensor<i64>, column: i64,
) {
    let row = block_id() * 256usize + UNIT_POS as usize;
    if row < rank.len() {
        if row == 0usize { counts[0] = 0i64; }
        let mut count = 0usize;
        if info[0] == 0i32 {
            let frozen = rank[row] >= 0i64;
            let active = !frozen && coefficient(pointers, columns, values, row, column) != F::new(0.0);
            count = merged_pattern(pointers, columns, row, pivots[column as usize] as usize, column,
                active, frozen, counts, 0usize, false);
        }
        counts[row + 1usize] = i64::cast_from(count);
    }
}

#[ruda(launch)]
fn emit_rows<F: Float + RudaElement>(
    pointers: &Tensor<i64>, columns: &Tensor<i64>, values: &Tensor<F>,
    rank: &Tensor<i64>, pivots: &Tensor<i64>, output_pointers: &Tensor<i64>, output: &mut Tensor<i64>, column: i64,
) {
    let row = block_id() * 256usize + UNIT_POS as usize;
    if row < rank.len() {
        let frozen = rank[row] >= 0i64;
        let active = !frozen && coefficient(pointers, columns, values, row, column) != F::new(0.0);
        merged_pattern(pointers, columns, row, pivots[column as usize] as usize, column,
            active, frozen, output, output_pointers[row] as usize, true);
    }
}

#[ruda(launch)]
fn eliminate<F: Float + RudaElement>(
    pointers: &Tensor<i64>, columns: &Tensor<i64>, values: &Tensor<F>, rank: &Tensor<i64>, pivots: &Tensor<i64>,
    rhs: &Tensor<F>, output_pointers: &Tensor<i64>, output_columns: &Tensor<i64>,
    output_values: &mut Tensor<F>, output_rhs: &mut Tensor<F>, column: i64,
) {
    let row = block_id();
    if row >= rank.len() { return; }
    let tid = UNIT_POS as usize;
    let pivot = pivots[column as usize] as usize;
    let frozen = rank[row] >= 0i64;
    let mut multiplier = F::new(0.0);
    let mut active = false;
    if !frozen {
        let a = coefficient(pointers, columns, values, row, column);
        if a != F::new(0.0) {
            multiplier = a / coefficient(pointers, columns, values, pivot, column);
            active = true;
        }
    }
    let begin = output_pointers[row] as usize;
    let end = output_pointers[row + 1usize] as usize;
    let mut offset = tid;
    let count = end - begin;
    let mut remaining = offset < count;
    while remaining {
        let p = begin + offset;
        let mut value = F::new(0.0);
        if frozen {
            value = values[pointers[row] as usize + offset];
        } else {
            let j = output_columns[p];
            value = coefficient(pointers, columns, values, row, j);
            if active {
                let pivot_end = pointers[pivot + 1usize] as usize;
                let q = lower(columns, pointers[pivot] as usize, pivot_end, j);
                if q < pivot_end {
                    if columns[q] == j { value = fma(-multiplier, values[q], value); }
                }
            }
        }
        output_values[p] = value;
        if count - offset <= 256usize { remaining = false; } else { offset += 256usize; }
    }
    if tid == 0usize {
        let mut value = rhs[row];
        if active { value = fma(-multiplier, rhs[pivot], value); }
        output_rhs[row] = value;
    }
}

#[ruda(launch)]
fn back_substitute<F: Float + RudaElement>(
    pointers: &Tensor<i64>, columns: &Tensor<i64>, values: &Tensor<F>,
    pivots: &Tensor<i64>, rhs: &Tensor<F>, output: &mut Tensor<F>,
) {
    let tid = UNIT_POS as usize;
    for step in 0..output.len() {
        let column = output.len() - 1usize - step;
        let row = pivots[column] as usize;
        let begin = pointers[row] as usize;
        let end = pointers[row + 1usize] as usize;
        let count = end - begin;
        let mut offset = tid;
        let mut active = offset < count;
        let mut sum = F::new(0.0);
        while active {
            let p = begin + offset;
            let j = columns[p];
            if j > i64::cast_from(column) { sum = fma(values[p], output[j as usize], sum); }
            if count - offset <= 256usize { active = false; } else { offset += 256usize; }
        }
        sum = crate::blas_accumulate::block_sum(sum);
        if tid == 0usize {
            output[column] = (rhs[row] - sum) / coefficient(pointers, columns, values, row, i64::cast_from(column));
        }
        sync_ruda();
    }
}

fn grid(blocks: usize) -> RudaCount {
    assert!(blocks > 0 && blocks <= u32::MAX as usize);
    let x = blocks.min(131072);
    RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1)
}

fn vector(v: &View, n: usize, dtype: u32) {
    assert_eq!(v.dtype, dtype);
    assert_eq!(v.shape, vec![n]);
    assert_eq!(v.strides, vec![1]);
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64]) {
    assert!(matches!(op, 542..=544));
    assert_eq!(xs.len(), if op == 542 { 7 } else if op == 543 { 10 } else { 6 });
    assert_eq!(xs[0].shape.len(), 1);
    assert!(xs[0].len > 1);
    let n = xs[0].len - 1;
    assert!(n <= i32::MAX as usize);
    let dtype = xs[2].dtype;
    assert!(matches!(dtype, 0 | 9));
    vector(&xs[0], n + 1, 4);
    vector(&xs[1], xs[2].len, 4);
    vector(&xs[2], xs[1].len, dtype);
    vector(&xs[3], n, 4);
    let c = client();
    let dim = RudaDim::new_1d(256);
    if op == 544 {
        assert!(ints.is_empty());
        vector(&xs[4], n, dtype);
        vector(&xs[5], n, dtype);
        macro_rules! run { ($f:ty) => { unsafe {
            back_substitute::launch::<$f, CudaRuntime>(&c, RudaCount::Static(1, 1, 1), dim,
                xs[0].arg(), xs[1].arg(), xs[2].arg(), xs[3].arg(), xs[4].arg(), xs[5].arg());
        }}; }
        if dtype == 0 { run!(f32); } else { run!(f64); }
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
    } else {
        assert_eq!(ints.len(), 1);
        assert!(ints[0] >= 0 && (ints[0] as usize) < n);
        vector(&xs[4], n, 4);
        if op == 542 {
            vector(&xs[5], 1, 5);
            vector(&xs[6], n + 1, 4);
            macro_rules! run { ($f:ty) => { unsafe {
                choose_pivot::launch::<$f, CudaRuntime>(&c, RudaCount::Static(1, 1, 1), dim,
                    xs[0].arg(), xs[1].arg(), xs[2].arg(), xs[3].arg(), xs[4].arg(), xs[5].arg(), ints[0]);
                count_rows::launch::<$f, CudaRuntime>(&c, grid(n.div_ceil(256)), dim,
                    xs[0].arg(), xs[1].arg(), xs[2].arg(), xs[3].arg(), xs[4].arg(), xs[5].arg(), xs[6].arg(), ints[0]);
            }}; }
            if dtype == 0 { run!(f32); } else { run!(f64); }
        } else {
            vector(&xs[5], n, dtype);
            vector(&xs[6], n + 1, 4);
            vector(&xs[7], xs[8].len, 4);
            vector(&xs[8], xs[7].len, dtype);
            vector(&xs[9], n, dtype);
            macro_rules! run { ($f:ty) => { unsafe {
                emit_rows::launch::<$f, CudaRuntime>(&c, grid(n.div_ceil(256)), dim,
                    xs[0].arg(), xs[1].arg(), xs[2].arg(), xs[3].arg(), xs[4].arg(), xs[6].arg(), xs[7].arg(), ints[0]);
                eliminate::launch::<$f, CudaRuntime>(&c, grid(n), dim,
                    xs[0].arg(), xs[1].arg(), xs[2].arg(), xs[3].arg(), xs[4].arg(), xs[5].arg(),
                    xs[6].arg(), xs[7].arg(), xs[8].arg(), xs[9].arg(), ints[0]);
            }}; }
            if dtype == 0 { run!(f32); } else { run!(f64); }
        }
        LAUNCHES.fetch_add(2, Ordering::Relaxed);
    }
    finish_dispatch(&c);
}
