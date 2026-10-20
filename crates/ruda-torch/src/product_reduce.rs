use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn source_address<S: Numeric>(input: &Tensor<S>, row: usize, item: usize, #[comptime] reduced: Vec<i64>) -> usize {
    let mut outer = row; let mut inner = item; let mut offset = 0usize;
    #[unroll]
    for axis in 0..comptime!(reduced.len()) {
        let dim = comptime!(reduced.len() - axis - 1); let mut coordinate = 0usize;
        if comptime!(reduced[dim] != 0) { coordinate = inner % input.shape(dim); inner /= input.shape(dim); }
        else { coordinate = outer % input.shape(dim); outer /= input.shape(dim); }
        offset += coordinate * input.stride(dim);
    }
    offset
}

#[ruda]
fn multiply<A: Numeric>(ar: A, ai: A, br: A, bi: A, #[comptime] code: u32) -> (A, A) {
    let mut real = A::cast_from(0u32); let mut imaginary = A::cast_from(0u32);
    if comptime!(code == 3) { real = A::cast_from(ar != A::cast_from(0u32) && br != A::cast_from(0u32)); }
    else if comptime!(code == 10 || code == 11 || code == 14) {
        real = ar * br - ai * bi; imaginary = ar * bi + ai * br;
    } else { real = ar * br; }
    (real, imaginary)
}

#[ruda]
fn block_product<A: Numeric>(real: A, imaginary: A, #[comptime] code: u32) -> (A, A) {
    let tid = UNIT_POS as usize; let mut re = real; let mut im = imaginary; let mut distance = 16u32;
    while distance > 0u32 {
        let (next_re, next_im) = multiply::<A>(re, im, plane_shuffle_down(re, distance), plane_shuffle_down(im, distance), code);
        re = next_re; im = next_im; distance /= 2u32;
    }
    let mut shared_re = SharedMemory::<A>::new(8usize); let mut shared_im = SharedMemory::<A>::new(8usize);
    sync_ruda();
    if tid % 32usize == 0usize {
        shared_re[tid / 32usize] = re;
        if comptime!(code == 10 || code == 11 || code == 14) { shared_im[tid / 32usize] = im; }
    }
    sync_ruda();
    re = A::cast_from(1u32); im = A::cast_from(0u32);
    if tid < 8usize {
        re = shared_re[tid];
        if comptime!(code == 10 || code == 11 || code == 14) { im = shared_im[tid]; }
    }
    if tid < 32usize {
        distance = 16u32;
        while distance > 0u32 {
            let (next_re, next_im) = multiply::<A>(re, im, plane_shuffle_down(re, distance), plane_shuffle_down(im, distance), code);
            re = next_re; im = next_im; distance /= 2u32;
        }
    }
    (re, im)
}

#[ruda(launch)]
fn empty<S: Numeric + RudaElement>(output: &mut Tensor<S>, imaginary: &mut Tensor<S>, #[comptime] code: u32) {
    let row = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if row < output.len() {
        let mut identity = S::cast_from(1u32);
        if comptime!(code == 12) { identity = S::cast_from(56u32); }
        else if comptime!(code == 13) { identity = S::cast_from(60u32); }
        output[address(output, row)] = identity;
        if comptime!(code == 10 || code == 11 || code == 14) { imaginary[address(imaginary, row)] = S::cast_from(0u32); }
    }
}

#[ruda(launch)]
fn small<S: Numeric + RudaElement, A: Numeric>(
    input: &Tensor<S>, imaginary: &Tensor<S>, output: &mut Tensor<S>, output_imaginary: &mut Tensor<S>,
    width: usize, #[comptime] reduced: Vec<i64>, #[comptime] code: u32,
) {
    let row = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if row < output.len() {
        let mut re = A::cast_from(1u32); let mut im = A::cast_from(0u32); let mut item = 0usize;
        while item < width {
            let offset = source_address::<S>(input, row, item, reduced.clone());
            let value = A::cast_from(input[offset]); let mut value_i = A::cast_from(0u32);
            if comptime!(code == 10 || code == 11 || code == 14) { value_i = A::cast_from(imaginary[offset]); }
            let (next_re, next_im) = multiply::<A>(re, im, value, value_i, code);
            re = next_re; im = next_im; item += 1usize;
        }
        output[address(output, row)] = S::cast_from(re);
        if comptime!(code == 10 || code == 11 || code == 14) { output_imaginary[address(output_imaginary, row)] = S::cast_from(im); }
    }
}

#[ruda(launch)]
fn chunks<S: Numeric + RudaElement, A: Numeric + RudaElement>(
    input: &Tensor<S>, imaginary: &Tensor<S>, partials: &mut Tensor<A>,
    output: &mut Tensor<S>, output_imaginary: &mut Tensor<S>, width: usize, count: usize,
    #[comptime] reduced: Vec<i64>, #[comptime] code: u32, #[comptime] direct: bool,
) {
    let task = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    let row = task / count; let chunk = task % count; let tid = UNIT_POS as usize;
    if row < output.len() {
        let base = chunk * 65536usize; let remaining = width - base;
        let mut re = A::cast_from(1u32); let mut im = A::cast_from(0u32); let mut item = tid;
        while item < remaining && item < 65536usize {
            let offset = source_address::<S>(input, row, base + item, reduced.clone());
            let value = A::cast_from(input[offset]); let mut value_i = A::cast_from(0u32);
            if comptime!(code == 10 || code == 11 || code == 14) { value_i = A::cast_from(imaginary[offset]); }
            let (next_re, next_im) = multiply::<A>(re, im, value, value_i, code);
            re = next_re; im = next_im; item += 256usize;
        }
        let (value, value_i) = block_product::<A>(re, im, code);
        if tid == 0usize {
            if comptime!(direct) {
                output[address(output, row)] = S::cast_from(value);
                if comptime!(code == 10 || code == 11 || code == 14) { output_imaginary[address(output_imaginary, row)] = S::cast_from(value_i); }
            } else {
                partials[task] = value;
                if comptime!(code == 10 || code == 11 || code == 14) { partials[output.len() * count + task] = value_i; }
            }
        }
    }
}

#[ruda(launch)]
fn finish<S: Numeric + RudaElement, A: Numeric + RudaElement>(
    partials: &Tensor<A>, output: &mut Tensor<S>, imaginary: &mut Tensor<S>, count: usize, #[comptime] code: u32,
) {
    let row = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize; let tid = UNIT_POS as usize;
    if row < output.len() {
        let mut re = A::cast_from(1u32); let mut im = A::cast_from(0u32); let mut chunk = tid;
        while chunk < count {
            let offset = row * count + chunk; let value = partials[offset]; let mut value_i = A::cast_from(0u32);
            if comptime!(code == 10 || code == 11 || code == 14) { value_i = partials[output.len() * count + offset]; }
            let (next_re, next_im) = multiply::<A>(re, im, value, value_i, code);
            re = next_re; im = next_im; chunk += 256usize;
        }
        let (value, value_i) = block_product::<A>(re, im, code);
        if tid == 0usize {
            output[address(output, row)] = S::cast_from(value);
            if comptime!(code == 10 || code == 11 || code == 14) { imaginary[address(imaginary, row)] = S::cast_from(value_i); }
        }
    }
}

fn grid(blocks: usize) -> RudaCount {
    assert!(blocks > 0 && blocks <= u32::MAX as usize, "product reduction launch exceeds U32 indexing");
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
    assert_eq!(input.dtype, output.dtype); assert!(input.dtype <= 14);
    assert_eq!(ints.len(), input.shape.len()); assert!(ints.iter().all(|&dim| matches!(dim, 0 | 1)));
    let extent = |reduce: bool| input.shape.iter().enumerate().filter(|(dim, _)| (ints[*dim] != 0) == reduce)
        .try_fold(1usize, |n, (_, &size)| n.checked_mul(size)).filter(|&n| n <= u32::MAX as usize)
        .expect("product reduction extent exceeds U32 indexing");
    let rows = extent(false); let width = extent(true); assert_eq!(output.len, rows);
    if rows == 0 { return; }
    let c = client(); let block = RudaDim::new_1d(256);
    let out_re = part(output, false); let out_im = part(output, true);
    if width == 0 {
        macro_rules! launch { ($s:ty) => { unsafe { empty::launch::<$s, CudaRuntime>(&c, grid(rows.div_ceil(256)), block,
            out_re.arg(), out_im.arg(), output.dtype); }}; }
        match out_re.dtype {
            0 => launch!(f32), 1 => launch!(f16), 2 => launch!(bf16), 3 | 8 | 12 | 13 => launch!(u8),
            4 => launch!(i64), 5 => launch!(i32), 6 => launch!(i16), 7 => launch!(i8), 9 => launch!(f64), _ => unreachable!(),
        }
        LAUNCHES.fetch_add(1, Ordering::Relaxed); finish_dispatch(&c); return;
    }
    assert!(!matches!(input.dtype, 12 | 13), "prod does not dispatch nonempty float8 input without a supported output dtype");
    let re = part(input, false); let im = part(input, true); let reduced = ints.to_vec();
    if width <= 32 {
        macro_rules! launch { ($s:ty, $a:ty) => { unsafe { small::launch::<$s, $a, CudaRuntime>(&c, grid(rows.div_ceil(256)), block,
            re.arg(), im.arg(), out_re.arg(), out_im.arg(), width, reduced, input.dtype); }}; }
        match input.dtype {
            0 | 10 => launch!(f32, f32), 1 | 14 => launch!(f16, f32), 2 => launch!(bf16, f32),
            3 | 8 => launch!(u8, u64), 4 => launch!(i64, u64), 5 => launch!(i32, u64),
            6 => launch!(i16, u64), 7 => launch!(i8, u64), 9 | 11 => launch!(f64, f64), _ => unreachable!(),
        }
        LAUNCHES.fetch_add(1, Ordering::Relaxed); finish_dispatch(&c); return;
    }
    let count = width.div_ceil(65536); let complex = matches!(input.dtype, 10 | 11 | 14);
    let tasks = rows.checked_mul(count).expect("product reduction task count overflow");
    let partial_len = if count == 1 { 1 } else { tasks.checked_mul(if complex { 2 } else { 1 }).expect("product workspace length overflow") };
    assert!(partial_len <= u32::MAX as usize);
    let accumulator = if (3..=8).contains(&input.dtype) { 4 } else if matches!(input.dtype, 9 | 11) { 9 } else { 0 };
    let partials = View::packed(c.empty(partial_len.checked_mul(element_bytes(accumulator)).expect("product workspace byte count overflow")), vec![partial_len], accumulator);
    macro_rules! launch { ($s:ty, $a:ty) => { unsafe {
        chunks::launch::<$s, $a, CudaRuntime>(&c, grid(tasks), block, re.arg(), im.arg(), partials.arg(), out_re.arg(), out_im.arg(),
            width, count, reduced, input.dtype, count == 1);
        if count > 1 { finish::launch::<$s, $a, CudaRuntime>(&c, grid(rows), block, partials.arg(), out_re.arg(), out_im.arg(), count, input.dtype); }
    }}; }
    match input.dtype {
        0 | 10 => launch!(f32, f32), 1 | 14 => launch!(f16, f32), 2 => launch!(bf16, f32),
        3 | 8 => launch!(u8, u64), 4 => launch!(i64, u64), 5 => launch!(i32, u64),
        6 => launch!(i16, u64), 7 => launch!(i8, u64), 9 | 11 => launch!(f64, f64), _ => unreachable!(),
    }
    LAUNCHES.fetch_add(if count == 1 { 1 } else { 2 }, Ordering::Relaxed); finish_dispatch(&c);
}
