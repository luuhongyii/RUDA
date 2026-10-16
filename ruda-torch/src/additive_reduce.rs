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
fn add<A: Numeric>(a: A, b: A, #[comptime] code: u32) -> A {
    let mut result = A::cast_from(0u32);
    if comptime!(code == 3) { result = A::cast_from(a != A::cast_from(0u32) || b != A::cast_from(0u32)); }
    else { result = a + b; }
    result
}

#[ruda]
fn block_sum<A: Numeric>(value: A, #[comptime] code: u32) -> A {
    let tid = UNIT_POS as usize; let mut result = value; let mut distance = 16u32;
    while distance > 0u32 {
        result = add(result, plane_shuffle_down(result, distance), code); distance /= 2u32;
    }
    let mut shared = SharedMemory::<A>::new(8usize);
    sync_ruda();
    if tid % 32usize == 0usize { shared[tid / 32usize] = result; }
    sync_ruda();
    result = A::cast_from(0u32);
    if tid < 8usize { result = shared[tid]; }
    if tid < 32usize {
        distance = 16u32;
        while distance > 0u32 {
            result = add(result, plane_shuffle_down(result, distance), code); distance /= 2u32;
        }
    }
    result
}

#[ruda]
fn emit<O: Numeric, A: Numeric>(
    output: &mut Tensor<O>, imaginary: &mut Tensor<O>, row: usize, real: A, imag: A, factor: A,
    #[comptime] mean: bool, #[comptime] code: u32,
) {
    let mut re = real; let mut im = imag;
    if comptime!(mean) { re = re * factor; im = im * factor; }
    output[address(output, row)] = O::cast_from(re);
    if comptime!(code == 10 || code == 11 || code == 14) { imaginary[address(imaginary, row)] = O::cast_from(im); }
}

#[ruda(launch)]
fn small<S: Numeric + RudaElement, O: Numeric + RudaElement, A: Numeric + RudaElement>(
    input: &Tensor<S>, imaginary: &Tensor<S>, output: &mut Tensor<O>, output_imaginary: &mut Tensor<O>,
    width: usize, factor: A, #[comptime] reduced: Vec<i64>, #[comptime] mean: bool, #[comptime] code: u32,
) {
    let row = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if row < output.len() {
        let mut re = A::cast_from(0u32); let mut im = A::cast_from(0u32); let mut item = 0usize;
        while item < width {
            let offset = source_address(input, row, item, reduced);
            re = add(re, A::cast_from(input[offset]), code);
            if comptime!(code == 10 || code == 11 || code == 14) { im = add(im, A::cast_from(imaginary[offset]), code); }
            item += 1usize;
        }
        emit(output, output_imaginary, row, re, im, factor, mean, code);
    }
}

#[ruda(launch)]
fn chunks<S: Numeric + RudaElement, O: Numeric + RudaElement, A: Numeric + RudaElement>(
    input: &Tensor<S>, imaginary: &Tensor<S>, partials: &mut Tensor<A>,
    output: &mut Tensor<O>, output_imaginary: &mut Tensor<O>, width: usize, count: usize, factor: A,
    #[comptime] reduced: Vec<i64>, #[comptime] mean: bool, #[comptime] code: u32, #[comptime] direct: bool,
) {
    let task = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    let row = task / count; let chunk = task % count; let tid = UNIT_POS as usize;
    if row < output.len() {
        let base = chunk * 65536usize; let remaining = width - base;
        let mut re = A::cast_from(0u32); let mut im = A::cast_from(0u32); let mut item = tid;
        while item < remaining && item < 65536usize {
            let offset = source_address(input, row, base + item, reduced);
            re = add(re, A::cast_from(input[offset]), code);
            if comptime!(code == 10 || code == 11 || code == 14) { im = add(im, A::cast_from(imaginary[offset]), code); }
            item += 256usize;
        }
        re = block_sum(re, code);
        if comptime!(code == 10 || code == 11 || code == 14) { im = block_sum(im, code); }
        if tid == 0usize {
            if comptime!(direct) { emit(output, output_imaginary, row, re, im, factor, mean, code); }
            else {
                partials[task] = re;
                if comptime!(code == 10 || code == 11 || code == 14) { partials[output.len() * count + task] = im; }
            }
        }
    }
}

#[ruda(launch)]
fn finish<O: Numeric + RudaElement, A: Numeric + RudaElement>(
    partials: &Tensor<A>, output: &mut Tensor<O>, imaginary: &mut Tensor<O>, count: usize, factor: A,
    #[comptime] mean: bool, #[comptime] code: u32,
) {
    let row = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize; let tid = UNIT_POS as usize;
    if row < output.len() {
        let mut re = A::cast_from(0u32); let mut im = A::cast_from(0u32); let mut chunk = tid;
        while chunk < count {
            let offset = row * count + chunk; re = add(re, partials[offset], code);
            if comptime!(code == 10 || code == 11 || code == 14) { im = add(im, partials[output.len() * count + offset], code); }
            chunk += 256usize;
        }
        re = block_sum(re, code);
        if comptime!(code == 10 || code == 11 || code == 14) { im = block_sum(im, code); }
        if tid == 0usize { emit(output, imaginary, row, re, im, factor, mean, code); }
    }
}

fn grid(blocks: usize) -> RudaCount {
    assert!(blocks > 0 && blocks <= u32::MAX as usize, "sum/mean launch exceeds U32 indexing");
    let x = blocks.min(131072);
    RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1)
}

fn part(input: &View, imaginary: bool) -> View {
    if matches!(input.dtype, 10 | 11 | 14) {
        let result = scientific::component(input, imaginary); scientific::indexable(&result); result
    } else { View { handle: input.handle.clone(), shape: input.shape.clone(), strides: input.strides.clone(), len: input.len, dtype: input.dtype } }
}

pub(super) fn dispatch(xs: &[View], ints: &[i64]) {
    assert_eq!(xs.len(), 2); let input = &xs[0]; let output = &xs[1];
    assert_eq!(ints.len(), input.shape.len() + 1); assert!(ints.iter().all(|&v| matches!(v, 0 | 1)));
    let mean = ints[0] != 0; let reduced = ints[1..].to_vec(); let code = output.dtype;
    assert!(input.dtype == code || code == 0 && matches!(input.dtype, 1 | 2));
    assert!(code <= 11 || code == 14, "nonempty sum/mean requires a supported arithmetic output dtype");
    assert!(!mean || !matches!(code, 3 | 14), "mean does not dispatch nonempty bool or complex32 output");
    let extent = |reduce: bool| input.shape.iter().enumerate().filter(|(dim, _)| (reduced[*dim] != 0) == reduce)
        .try_fold(1usize, |n, (_, &size)| n.checked_mul(size)).filter(|&n| n <= u32::MAX as usize)
        .expect("sum/mean extent exceeds U32 indexing");
    let rows = extent(false); let width = extent(true); assert_eq!(output.len, rows);
    if rows == 0 { return; }
    assert!(width > 0, "sum/mean native reduction dimension must be nonempty");
    let c = client(); let block = RudaDim::new_1d(256);
    let re = part(input, false); let im = part(input, true); let out_re = part(output, false); let out_im = part(output, true);
    let factor32 = rows as f32 / input.len as f32; let factor64 = rows as f64 / input.len as f64;
    let numerator = match code { 5 => rows as i32 as i64, 6 => rows as i16 as i64, 7 => rows as i8 as i64, 8 => rows as u8 as i64, _ => rows as i64 };
    let integer_factor = (numerator / input.len as i64) as u64;
    let count = width.div_ceil(65536); let complex = matches!(code, 10 | 11 | 14);
    let tasks = rows.checked_mul(count).expect("sum/mean task count overflow");
    let accumulator = if (3..=8).contains(&code) { 4 } else if matches!(code, 9 | 11) { 9 } else { 0 };
    let workspace = if width > 32 {
        let length = if count == 1 { 1 } else { tasks.checked_mul(if complex { 2 } else { 1 }).expect("sum/mean workspace length overflow") };
        assert!(length <= u32::MAX as usize);
        Some(View::packed(c.empty(length.checked_mul(element_bytes(accumulator)).expect("sum/mean workspace byte count overflow")), vec![length], accumulator))
    } else { None };
    macro_rules! launch { ($s:ty, $o:ty, $a:ty, $factor:expr) => { unsafe {
        if width <= 32 {
            small::launch::<$s, $o, $a, CudaRuntime>(&c, grid(rows.div_ceil(256)), block,
                re.arg(), im.arg(), out_re.arg(), out_im.arg(), width, $factor, reduced, mean, code);
        } else {
            let partials = workspace.as_ref().unwrap();
            chunks::launch::<$s, $o, $a, CudaRuntime>(&c, grid(tasks), block,
                re.arg(), im.arg(), partials.arg(), out_re.arg(), out_im.arg(), width, count, $factor, reduced, mean, code, count == 1);
            if count > 1 { finish::launch::<$o, $a, CudaRuntime>(&c, grid(rows), block, partials.arg(), out_re.arg(), out_im.arg(), count, $factor, mean, code); }
        }
    }}; }
    match code {
        0 | 10 => match input.dtype {
            1 => launch!(f16, f32, f32, factor32), 2 => launch!(bf16, f32, f32, factor32), _ => launch!(f32, f32, f32, factor32),
        },
        1 | 14 => launch!(f16, f16, f32, factor32), 2 => launch!(bf16, bf16, f32, factor32),
        3 | 8 => launch!(u8, u8, u64, integer_factor), 4 => launch!(i64, i64, u64, integer_factor),
        5 => launch!(i32, i32, u64, integer_factor), 6 => launch!(i16, i16, u64, integer_factor),
        7 => launch!(i8, i8, u64, integer_factor), 9 | 11 => launch!(f64, f64, f64, factor64), _ => unreachable!(),
    }
    LAUNCHES.fetch_add(if count == 1 || width <= 32 { 1 } else { 2 }, Ordering::Relaxed); finish_dispatch(&c);
}
