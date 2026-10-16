use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn source_address<S: Numeric>(input: &Tensor<S>, row: usize, item: usize, #[comptime] reduced: Vec<i64>) -> usize {
    let mut outer = row; let mut inner = item; let mut offset = 0usize;
    #[unroll]
    for axis in 0..comptime!(reduced.len()) {
        let dim = comptime!(reduced.len() - axis - 1);
        let mut coordinate = 0usize;
        if comptime!(reduced[dim] != 0) { coordinate = inner % input.shape(dim); inner /= input.shape(dim); }
        else { coordinate = outer % input.shape(dim); outer /= input.shape(dim); }
        offset += coordinate * input.stride(dim);
    }
    offset
}

#[ruda]
fn truth<S: Numeric>(input: &Tensor<S>, imaginary: &Tensor<S>, offset: usize, #[comptime] code: u32) -> u32 {
    let value = input[offset]; let mut nonzero = value != S::cast_from(0u32);
    if comptime!(code == 12 || code == 13) { nonzero = (u8::cast_from(value) & 127u8) != 0u8; }
    if comptime!(code == 10 || code == 11 || code == 14) { nonzero = nonzero || imaginary[offset] != S::cast_from(0u32); }
    u32::cast_from(nonzero)
}

#[ruda]
fn combine(a: u32, b: u32, #[comptime] mode: u32) -> u32 {
    let mut result = 0u32;
    if comptime!(mode == 0) { result = u32::cast_from(a != 0u32 && b != 0u32); }
    else if comptime!(mode == 1) { result = u32::cast_from(a != 0u32 || b != 0u32); }
    else { result = a + b; }
    result
}

#[ruda]
fn block_reduce(value: u32, #[comptime] mode: u32) -> u32 {
    let tid = UNIT_POS as usize; let mut result = value; let mut distance = 16u32;
    while distance > 0u32 {
        result = combine(result, plane_shuffle_down(result, distance), mode); distance /= 2u32;
    }
    let mut shared = SharedMemory::<u32>::new(8usize);
    sync_ruda();
    if tid % 32usize == 0usize { shared[tid / 32usize] = result; }
    sync_ruda();
    result = u32::cast_from(comptime!(mode == 0));
    if tid < 8usize { result = shared[tid]; }
    if tid < 32usize {
        distance = 16u32;
        while distance > 0u32 {
            result = combine(result, plane_shuffle_down(result, distance), mode); distance /= 2u32;
        }
    }
    result
}

#[ruda(launch)]
fn empty<O: Numeric + RudaElement>(output: &mut Tensor<O>, #[comptime] mode: u32) {
    let row = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if row < output.len() { output[address(output, row)] = O::cast_from(comptime!(mode == 0)); }
}

#[ruda(launch)]
fn small<S: Numeric + RudaElement, O: Numeric + RudaElement>(
    input: &Tensor<S>, imaginary: &Tensor<S>, output: &mut Tensor<O>, width: usize,
    #[comptime] reduced: Vec<i64>, #[comptime] mode: u32, #[comptime] code: u32,
) {
    let row = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if row < output.len() {
        let mut result = u32::cast_from(comptime!(mode == 0)); let mut item = 0usize;
        while item < width {
            let offset = source_address::<S>(input, row, item, reduced.clone());
            result = combine(result, truth::<S>(input, imaginary, offset, code), mode); item += 1usize;
        }
        output[address(output, row)] = O::cast_from(result);
    }
}

#[ruda(launch)]
fn chunks<S: Numeric + RudaElement, O: Numeric + RudaElement>(
    input: &Tensor<S>, imaginary: &Tensor<S>, partials: &mut Tensor<u32>, output: &mut Tensor<O>,
    width: usize, count: usize, #[comptime] reduced: Vec<i64>,
    #[comptime] mode: u32, #[comptime] code: u32, #[comptime] direct: bool,
) {
    let task = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    let row = task / count; let chunk = task % count; let tid = UNIT_POS as usize;
    if row < output.len() {
        let base = chunk * 65536usize; let remaining = width - base;
        let mut item = tid; let mut result = u32::cast_from(comptime!(mode == 0));
        while item < remaining && item < 65536usize {
            let offset = source_address::<S>(input, row, base + item, reduced.clone());
            result = combine(result, truth::<S>(input, imaginary, offset, code), mode); item += 256usize;
        }
        result = block_reduce(result, mode);
        if tid == 0usize {
            if comptime!(direct) { output[address(output, row)] = O::cast_from(result); }
            else { partials[task] = result; }
        }
    }
}

#[ruda(launch)]
fn finish<O: Numeric + RudaElement>(partials: &Tensor<u32>, output: &mut Tensor<O>, count: usize, #[comptime] mode: u32) {
    let row = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize; let tid = UNIT_POS as usize;
    if row < output.len() {
        let mut chunk = tid; let mut result = u32::cast_from(comptime!(mode == 0));
        while chunk < count {
            result = combine(result, partials[row * count + chunk], mode); chunk += 256usize;
        }
        result = block_reduce(result, mode);
        if tid == 0usize { output[address(output, row)] = O::cast_from(result); }
    }
}

fn grid(blocks: usize) -> RudaCount {
    assert!(blocks > 0 && blocks <= u32::MAX as usize, "logical reduction launch exceeds U32 indexing");
    let x = blocks.min(131072);
    RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1)
}

fn part(input: &View, imaginary: bool) -> View {
    if matches!(input.dtype, 10 | 11 | 14) {
        let result = scientific::component(input, imaginary); scientific::indexable(&result); result
    } else { View { handle: input.handle.clone(), shape: input.shape.clone(), strides: input.strides.clone(), len: input.len, dtype: input.dtype } }
}

pub(super) fn dispatch(xs: &[View], ints: &[i64]) {
    assert_eq!(xs.len(), 2);
    let input = &xs[0]; let output = &xs[1];
    assert_eq!(ints.len(), input.shape.len() + 1); assert!((0..=2).contains(&ints[0]));
    assert!(input.dtype <= 14 && ints[1..].iter().all(|&dim| matches!(dim, 0 | 1)));
    let mode = ints[0] as u32; let reduced = ints[1..].to_vec();
    if mode == 2 { assert_eq!(output.dtype, 4); } else { assert!(matches!(output.dtype, 3 | 8)); }
    let extent = |reduce: bool| input.shape.iter().enumerate().filter(|(dim, _)| (reduced[*dim] != 0) == reduce)
        .try_fold(1usize, |n, (_, &size)| n.checked_mul(size)).filter(|&n| n <= u32::MAX as usize)
        .expect("logical reduction extent exceeds U32 indexing");
    let rows = extent(false); let width = extent(true); assert_eq!(output.len, rows);
    if rows == 0 { return; }
    let c = client(); let block = RudaDim::new_1d(256);
    if width == 0 {
        macro_rules! launch { ($o:ty) => { unsafe { empty::launch::<$o, CudaRuntime>(&c, grid(rows.div_ceil(256)), block, output.arg(), mode); }}; }
        if mode == 2 { launch!(i64); } else { launch!(u8); }
        LAUNCHES.fetch_add(1, Ordering::Relaxed); finish_dispatch(&c); return;
    }
    let re = part(input, false); let im = part(input, true);
    if width <= 32 {
        macro_rules! launch { ($s:ty, $o:ty) => { unsafe { small::launch::<$s, $o, CudaRuntime>(&c, grid(rows.div_ceil(256)), block,
            re.arg(), im.arg(), output.arg(), width, reduced, mode, input.dtype); }}; }
        macro_rules! data { ($s:ty) => { if mode == 2 { launch!($s, i64); } else { launch!($s, u8); } }; }
        match re.dtype {
            0 => data!(f32), 1 => data!(f16), 2 => data!(bf16), 3 | 8 | 12 | 13 => data!(u8),
            4 => data!(i64), 5 => data!(i32), 6 => data!(i16), 7 => data!(i8), 9 => data!(f64), _ => unreachable!(),
        }
        LAUNCHES.fetch_add(1, Ordering::Relaxed); finish_dispatch(&c); return;
    }
    let count = width.div_ceil(65536);
    let tasks = rows.checked_mul(count).filter(|&n| n <= u32::MAX as usize).expect("logical reduction partial count overflow");
    let partial_len = if count == 1 { 1 } else { tasks };
    let partials = View::packed(c.empty(partial_len.checked_mul(4).expect("logical reduction workspace overflow")), vec![partial_len], 5);
    macro_rules! launch { ($s:ty, $o:ty) => { unsafe {
        chunks::launch::<$s, $o, CudaRuntime>(&c, grid(tasks), block, re.arg(), im.arg(), partials.arg(), output.arg(),
            width, count, reduced, mode, input.dtype, count == 1);
        if count > 1 { finish::launch::<$o, CudaRuntime>(&c, grid(rows), block, partials.arg(), output.arg(), count, mode); }
    }}; }
    macro_rules! data { ($s:ty) => { if mode == 2 { launch!($s, i64); } else { launch!($s, u8); } }; }
    match re.dtype {
        0 => data!(f32), 1 => data!(f16), 2 => data!(bf16), 3 | 8 | 12 | 13 => data!(u8),
        4 => data!(i64), 5 => data!(i32), 6 => data!(i16), 7 => data!(i8), 9 => data!(f64), _ => unreachable!(),
    }
    LAUNCHES.fetch_add(if count == 1 { 1 } else { 2 }, Ordering::Relaxed); finish_dispatch(&c);
}
