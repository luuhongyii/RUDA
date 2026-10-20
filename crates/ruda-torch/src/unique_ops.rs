use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn load<T: Numeric>(input: &Tensor<T>, p: usize, #[comptime] boolean: bool) -> T {
    let mut value = input[address(input, p)];
    if comptime!(boolean) { value = T::cast_from(value != T::cast_from(0u32)); }
    value
}

#[ruda]
fn float_key<T: Numeric>(value: T, #[comptime] code: u32) -> u64 {
    let mut bits = 0u64;
    let mut sign = 0x8000u64;
    let mut mask = 0xffffu64;
    if comptime!(code == 9) {
        bits = u64::reinterpret(value); sign = 0x8000000000000000u64; mask = u64::MAX.runtime();
    } else if comptime!(code == 0) {
        bits = u64::cast_from(u32::reinterpret(value)); sign = 0x80000000u64; mask = 0xffffffffu64;
    } else { bits = u64::cast_from(u16::reinterpret(value)); }
    if value == T::cast_from(0u32) { bits = 0u64; }
    let mut result = bits ^ sign;
    if bits & sign != 0u64 { result = bits ^ mask; }
    result
}

#[ruda]
fn before<T: Numeric>(input: &Tensor<T>, a: u32, b: u32, width: usize, #[comptime] code: u32, #[comptime] flat: bool) -> bool {
    let mut result = a < b;
    if comptime!(flat && (code == 0 || code == 1 || code == 2 || code == 9)) {
        let ka = float_key::<T>(load::<T>(input, a as usize, false), code);
        let kb = float_key::<T>(load::<T>(input, b as usize, false), code);
        if ka != kb { result = ka < kb; }
    } else {
        let mut column = 0usize;
        let mut decided = false;
        while column < width && !decided {
            let x = load::<T>(input, a as usize * width + column, code == 3);
            let y = load::<T>(input, b as usize * width + column, code == 3);
            let xn = x != x; let yn = y != y;
            if xn != yn { result = yn; decided = true; }
            else if !xn {
                if x < y { result = true; decided = true; }
                else if x > y { result = false; decided = true; }
            }
            column += 1usize;
        }
    }
    result
}

#[ruda(launch)]
fn initialize(order: &mut Tensor<u32>) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < order.len() { order[p] = u32::cast_from(p); }
}

#[ruda(launch)]
fn merge<T: Numeric + RudaElement>(
    input: &Tensor<T>, order: &Tensor<u32>, output: &mut Tensor<u32>, width: usize, run: usize,
    #[comptime] code: u32, #[comptime] flat: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    let length = order.len();
    if p < length {
        let group = p / run;
        let start = group * run;
        let mut other_start = start + usize::min(run, length - start);
        let mut other_end = other_start + usize::min(run, length - other_start);
        let mut base = start;
        if group % 2usize != 0 { other_start = start - run; other_end = start; base = other_start; }
        let key = order[p];
        let mut lo = other_start;
        let mut hi = other_end;
        while lo < hi {
            let mid = lo + (hi - lo) / 2usize;
            if before::<T>(input, order[mid], key, width, code, flat) { lo = mid + 1usize; }
            else { hi = mid; }
        }
        output[base + (p - start) + (lo - other_start)] = key;
    }
}

#[ruda(launch)]
fn heads<T: Numeric + RudaElement>(
    input: &Tensor<T>, order: &Tensor<u32>, groups: &mut Tensor<u32>, width: usize,
    #[comptime] consecutive: bool, #[comptime] boolean: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < groups.len() {
        let mut different = p == 0usize;
        if p > 0usize {
            let mut a = p - 1usize; let mut b = p;
            if comptime!(!consecutive) { a = order[p - 1usize] as usize; b = order[p] as usize; }
            let mut column = 0usize;
            while column < width && !different {
                different = load::<T>(input, a * width + column, boolean) != load::<T>(input, b * width + column, boolean);
                column += 1usize;
            }
        }
        groups[p] = u32::cast_from(different);
    }
}

#[ruda]
fn warp_prefix(value: u32) -> u32 {
    let lane = UNIT_POS % 32u32;
    let mut result = value;
    let mut step = 1u32;
    while step < 32u32 {
        let previous = plane_shuffle_up(result, step);
        if lane >= step { result += previous; }
        step *= 2u32;
    }
    result
}

#[ruda(launch)]
fn scan_tiles(groups: &mut Tensor<u32>, totals: &mut Tensor<u32>) {
    let block = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    let tid = UNIT_POS as usize;
    let p = block * 256usize + tid;
    if block < totals.len() {
        let mut value = 0u32;
        if p < groups.len() { value = groups[p]; }
        value = warp_prefix(value);
        let mut warps = SharedMemory::<u32>::new(8usize);
        if tid % 32usize == 31usize { warps[tid / 32usize] = value; }
        sync_ruda();
        if tid < 32usize {
            let mut subtotal = 0u32;
            if tid < 8usize { subtotal = warps[tid]; }
            subtotal = warp_prefix(subtotal);
            if tid < 8usize { warps[tid] = subtotal; }
        }
        sync_ruda();
        if tid >= 32usize { value += warps[tid / 32usize - 1usize]; }
        if p < groups.len() { groups[p] = value; }
        if tid == 255usize { totals[block] = value; }
    }
}

#[ruda(launch)]
fn add_offsets(groups: &mut Tensor<u32>, totals: &Tensor<u32>) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < groups.len() && p >= 256usize { groups[p] += totals[p / 256usize - 1usize]; }
}

#[ruda(launch)]
fn starts_and_size(groups: &Tensor<u32>, starts: &mut Tensor<u32>, size: &mut Tensor<i64>) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < groups.len() {
        let group = groups[p];
        let mut first = p == 0usize;
        if p > 0usize { first = group != groups[p - 1usize]; }
        if first { starts[group as usize - 1usize] = u32::cast_from(p); }
        if p == groups.len() - 1usize { size[0] = i64::cast_from(group); }
    }
}

#[ruda(launch)]
fn emit<T: Numeric + RudaElement>(
    input: &Tensor<T>, order: &Tensor<u32>, groups: &Tensor<u32>, starts: &Tensor<u32>,
    output: &mut Tensor<T>, inverse: &mut Tensor<i64>, counts: &mut Tensor<i64>, width: usize,
    #[comptime] consecutive: bool, #[comptime] return_inverse: bool, #[comptime] return_counts: bool, #[comptime] boolean: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < output.len() {
        let row = p / width;
        let column = p % width;
        let mut source = starts[row] as usize;
        if comptime!(!consecutive) { source = order[source] as usize; }
        output[p] = load::<T>(input, source * width + column, boolean);
    }
    if comptime!(return_inverse) {
        if p < groups.len() {
            let mut position = p;
            if comptime!(!consecutive) { position = order[p] as usize; }
            inverse[position] = i64::cast_from(groups[p] - 1u32);
        }
    }
    if comptime!(return_counts) {
        if p < counts.len() {
            let mut end = groups.len();
            if p + 1usize < counts.len() { end = starts[p + 1usize] as usize; }
            counts[p] = i64::cast_from(end - starts[p] as usize);
        }
    }
}

fn grid(blocks: usize) -> RudaCount {
    assert!(blocks > 0 && blocks <= u32::MAX as usize, "unique launch exceeds U32 indexing");
    let x = blocks.min(131072);
    RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1)
}

fn packed(value: &View, len: usize, code: u32) {
    assert_eq!(value.dtype, code); assert_eq!(value.shape, vec![len]); assert_eq!(value.strides, vec![1usize]);
}

fn scratch(c: &ComputeClient<CudaRuntime>, len: usize) -> View {
    View::packed(c.empty(len.checked_mul(4).expect("unique workspace overflow")), vec![len], 5)
}

pub(super) fn inclusive_prefix(c: &ComputeClient<CudaRuntime>, groups: &View) -> (Vec<View>, u64) {
    assert!(groups.len > 0);
    packed(groups, groups.len, 5);
    let block = RudaDim::new_1d(256);
    let mut levels = Vec::new();
    let mut items = groups.len;
    let mut launches = 0u64;
    loop {
        let blocks = items.div_ceil(256);
        levels.push(scratch(c, blocks));
        let level = levels.len() - 1;
        let values = if level == 0 { groups } else { &levels[level - 1] };
        unsafe { scan_tiles::launch::<CudaRuntime>(c, grid(blocks), block, values.arg(), levels[level].arg()); }
        launches += 1;
        if blocks == 1 { break; }
        items = blocks;
    }
    for level in (0..levels.len() - 1).rev() {
        let values = if level == 0 { groups } else { &levels[level - 1] };
        unsafe { add_offsets::launch::<CudaRuntime>(c, grid(values.len.div_ceil(256)), block, values.arg(), levels[level].arg()); }
        launches += 1;
    }
    (levels, launches)
}

pub(super) fn stable_order(c: &ComputeClient<CudaRuntime>, input: &View, order: &View, width: usize, flat: bool) -> (View, u64) {
    let length = order.len;
    assert!(length > 0 && width > 0 && input.dtype <= 9);
    assert_eq!(input.len, length.checked_mul(width).expect("sort input shape overflow"));
    packed(order, length, 5);
    let temporary = scratch(c, length);
    let point_grid = grid(length.div_ceil(256)); let block = RudaDim::new_1d(256);
    let passes = (usize::BITS - (length - 1).leading_zeros()) as usize;
    let mut current = if passes % 2 == 0 { order } else { &temporary };
    let mut next = if passes % 2 == 0 { &temporary } else { order };
    unsafe { initialize::launch::<CudaRuntime>(c, point_grid.clone(), block, current.arg()); }
    let mut launches = 1u64;
    let mut run = 1usize;
    while run < length {
        macro_rules! merge_rows { ($t:ty) => { unsafe { merge::launch::<$t, CudaRuntime>(c, point_grid.clone(), block,
            input.arg(), current.arg(), next.arg(), width, run, input.dtype, flat); }}; }
        match input.dtype {
            0 => merge_rows!(f32), 1 => merge_rows!(f16), 2 => merge_rows!(bf16), 3 | 8 => merge_rows!(u8),
            4 => merge_rows!(i64), 5 => merge_rows!(i32), 6 => merge_rows!(i16), 7 => merge_rows!(i8), 9 => merge_rows!(f64), _ => unreachable!(),
        }
        std::mem::swap(&mut current, &mut next);
        run = run.checked_mul(2).expect("sort stage overflow");
        launches += 1;
    }
    (temporary, launches)
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64]) {
    assert!(matches!(op, 461 | 462));
    assert_eq!(ints.len(), 5);
    assert!(ints[0] > 0 && ints[1] > 0 && ints[2..].iter().all(|&x| matches!(x, 0 | 1)));
    let length = ints[0] as usize; let width = ints[1] as usize; let consecutive = ints[2] != 0;
    assert!(length <= u32::MAX as usize && width <= u32::MAX as usize);
    assert_eq!(xs.len(), if op == 461 { 5 } else { 7 });
    let input = &xs[0]; let order = &xs[1]; let groups = &xs[2]; let starts = &xs[3];
    assert!(input.dtype <= 9);
    assert_eq!(input.len, length.checked_mul(width).expect("unique input shape overflow"));
    packed(order, if consecutive { 0 } else { length }, 5); packed(groups, length, 5); packed(starts, length, 5);
    let c = client(); let block = RudaDim::new_1d(256);
    let point_grid = grid(length.div_ceil(256));
    if op == 461 {
        let flat = ints[3] != 0;
        assert_eq!(ints[4], 0);
        if flat { assert_eq!(width, 1); }
        packed(&xs[4], 1, 4);
        let mut launches = 0u64;
        let _sort_workspace = if consecutive { None } else {
            let (temporary, sort_launches) = stable_order(&c, input, order, width, flat);
            launches += sort_launches;
            Some(temporary)
        };
        macro_rules! mark { ($t:ty) => { unsafe { heads::launch::<$t, CudaRuntime>(&c, point_grid.clone(), block,
            input.arg(), order.arg(), groups.arg(), width, consecutive, input.dtype == 3); }}; }
        match input.dtype { 0 => mark!(f32), 1 => mark!(f16), 2 => mark!(bf16), 3 | 8 => mark!(u8),
            4 => mark!(i64), 5 => mark!(i32), 6 => mark!(i16), 7 => mark!(i8), 9 => mark!(f64), _ => unreachable!() }
        launches += 1;
        let (_levels, scans) = inclusive_prefix(&c, groups);
        launches += scans;
        unsafe { starts_and_size::launch::<CudaRuntime>(&c, point_grid, block, groups.arg(), starts.arg(), xs[4].arg()); }
        LAUNCHES.fetch_add(launches + 1, Ordering::Relaxed);
        finish_dispatch(&c);
    } else {
        let return_inverse = ints[3] != 0; let return_counts = ints[4] != 0;
        let output = &xs[4]; let inverse = &xs[5]; let counts = &xs[6];
        assert!(output.len > 0 && output.len % width == 0 && output.len <= input.len);
        let size = output.len / width;
        packed(output, output.len, input.dtype); packed(inverse, if return_inverse { length } else { 0 }, 4);
        packed(counts, if return_counts { size } else { 0 }, 4);
        let work = output.len.max(inverse.len);
        macro_rules! write { ($t:ty) => { unsafe { emit::launch::<$t, CudaRuntime>(&c, grid(work.div_ceil(256)), block,
            input.arg(), order.arg(), groups.arg(), starts.arg(), output.arg(), inverse.arg(), counts.arg(), width,
            consecutive, return_inverse, return_counts, input.dtype == 3); }}; }
        match input.dtype { 0 => write!(f32), 1 => write!(f16), 2 => write!(bf16), 3 | 8 => write!(u8),
            4 => write!(i64), 5 => write!(i32), 6 => write!(i16), 7 => write!(i8), 9 => write!(f64), _ => unreachable!() }
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
        finish_dispatch(&c);
    }
}
