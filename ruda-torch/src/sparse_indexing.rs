use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn position() -> usize {
    (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize
}

#[ruda(launch)]
fn matches(sorted: &Tensor<i64>, queries: &Tensor<i64>, first: &mut Tensor<i64>, counts: &mut Tensor<i64>, size: i64) {
    let p = position();
    if p == 0usize { counts[address(counts, 0usize)] = 0i64; }
    if p < queries.len() {
        let mut query = queries[address(queries, p)]; if query < 0i64 { query += size; }
        let mut low = 0usize; let mut high = sorted.len();
        while low < high {
            let middle = low + (high - low) / 2usize;
            if sorted[address(sorted, middle)] < query { low = middle + 1usize; } else { high = middle; }
        }
        let begin = low; high = sorted.len();
        while low < high {
            let middle = low + (high - low) / 2usize;
            if sorted[address(sorted, middle)] <= query { low = middle + 1usize; } else { high = middle; }
        }
        first[address(first, p)] = i64::cast_from(begin);
        counts[address(counts, p + 1usize)] = i64::cast_from(low - begin);
    }
}

#[ruda(launch)]
fn emit(
    indices: &Tensor<i64>, first: &Tensor<i64>, offsets: &Tensor<i64>, order: &Tensor<i64>,
    output_indices: &mut Tensor<i64>, selected: &mut Tensor<i64>, #[comptime] axis: usize, #[comptime] identity: bool,
) {
    let p = position();
    if p < selected.len() {
        let mut low = 0usize; let mut high = first.len();
        while low < high {
            let middle = low + (high - low) / 2usize;
            if offsets[address(offsets, middle + 1usize)] <= i64::cast_from(p) { low = middle + 1usize; }
            else { high = middle; }
        }
        let query = low;
        let rank = usize::cast_from(first[address(first, query)] + i64::cast_from(p) - offsets[address(offsets, query)]);
        let mut source = rank; if comptime!(!identity) { source = usize::cast_from(order[address(order, rank)]); }
        selected[address(selected, p)] = i64::cast_from(source);
        for dim in 0..indices.shape(0) {
            let mut coordinate = i64::cast_from(query);
            if dim != axis { coordinate = indices[dim * indices.stride(0) + source * indices.stride(1)]; }
            output_indices[dim * output_indices.stride(0) + p * output_indices.stride(1)] = coordinate;
        }
    }
}

fn vector(v: &View) { assert_eq!(v.dtype, 4); assert_eq!(v.shape.len(), 1); }

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64]) {
    let c = client(); let dim = RudaDim::new_1d(256);
    let work;
    if op == 538 {
        assert_eq!(xs.len(), 4); assert_eq!(ints.len(), 1); assert!(ints[0] >= 0);
        for v in xs { vector(v); }
        assert_eq!(xs[2].len, xs[1].len); assert_eq!(xs[3].len, xs[1].len + 1);
        crate::indexed_write::validate(&c, &xs[1], usize::try_from(ints[0]).expect("sparse axis size overflow"), true);
        work = xs[1].len.max(1);
    } else {
        assert_eq!(op, 539); assert_eq!(xs.len(), 6); assert_eq!(ints.len(), 2);
        for v in [&xs[0], &xs[4]] { assert_eq!(v.dtype, 4); assert_eq!(v.shape.len(), 2); }
        for v in [&xs[1], &xs[2], &xs[3], &xs[5]] { vector(v); }
        assert!(ints[0] >= 0 && (ints[0] as usize) < xs[0].shape[0]); assert!(matches!(ints[1], 0 | 1));
        assert_eq!(xs[2].len, xs[1].len + 1);
        assert_eq!(xs[3].len, if ints[1] != 0 { 0 } else { xs[0].shape[1] });
        assert_eq!(xs[4].shape, [xs[0].shape[0], xs[5].len]);
        work = xs[5].len;
        if work == 0 { return; }
    }
    assert!(work <= u32::MAX as usize);
    let blocks = work.div_ceil(256); let x = blocks.min(131072);
    let grid = RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1);
    unsafe {
        if op == 538 {
            matches::launch::<CudaRuntime>(&c, grid, dim, xs[0].arg(), xs[1].arg(), xs[2].arg(), xs[3].arg(), ints[0]);
        } else {
            emit::launch::<CudaRuntime>(&c, grid, dim, xs[0].arg(), xs[1].arg(), xs[2].arg(), xs[3].arg(), xs[4].arg(), xs[5].arg(),
                ints[0] as usize, ints[1] != 0);
        }
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed); finish_dispatch(&c);
}
