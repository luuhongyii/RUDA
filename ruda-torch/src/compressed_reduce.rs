use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn combine<S: Numeric, A: Numeric>(
    ar: A, ai: A, br: A, bi: A, #[comptime] product: bool, #[comptime] complex: bool,
) -> (A, A) {
    let mut re = ar + br; let mut im = ai + bi;
    if comptime!(product) {
        if comptime!(complex) {
            re = A::cast_from(S::cast_from(ar * br - ai * bi));
            im = A::cast_from(S::cast_from(ar * bi + ai * br));
        } else {
            re = A::cast_from(S::cast_from(ar * br));
            im = A::cast_from(0u32);
        }
    }
    (re, im)
}

#[ruda]
fn block_reduce<S: Numeric, A: Numeric>(
    real: A, imaginary: A, #[comptime] product: bool, #[comptime] complex: bool,
) -> (A, A) {
    let tid = UNIT_POS as usize; let mut re = real; let mut im = imaginary; let mut distance = 16u32;
    while distance > 0u32 {
        let (next_re, next_im) = combine::<S, A>(re, im, plane_shuffle_down(re, distance),
            plane_shuffle_down(im, distance), product, complex);
        re = next_re; im = next_im; distance /= 2u32;
    }
    let mut shared_re = SharedMemory::<A>::new(8usize); let mut shared_im = SharedMemory::<A>::new(8usize);
    sync_ruda();
    if tid % 32usize == 0usize {
        shared_re[tid / 32usize] = re;
        if comptime!(complex) { shared_im[tid / 32usize] = im; }
    }
    sync_ruda();
    re = A::cast_from(0u32); im = A::cast_from(0u32);
    if comptime!(product) { re = A::cast_from(1u32); }
    if tid < 8usize {
        re = shared_re[tid];
        if comptime!(complex) { im = shared_im[tid]; }
    }
    if tid < 32usize {
        distance = 16u32;
        while distance > 0u32 {
            let (next_re, next_im) = combine::<S, A>(re, im, plane_shuffle_down(re, distance),
                plane_shuffle_down(im, distance), product, complex);
            re = next_re; im = next_im; distance /= 2u32;
        }
    }
    (re, im)
}

#[ruda(launch)]
fn segments<S: Numeric + RudaElement, A: Numeric, O: Numeric + RudaElement>(
    starts: &Tensor<i64>, ends: &Tensor<i64>, order: &Tensor<i64>,
    values: &Tensor<S>, imaginary: &Tensor<S>, output: &mut Tensor<O>, output_imaginary: &mut Tensor<O>,
    #[comptime] indirect: bool, #[comptime] product: bool, #[comptime] complex: bool,
) {
    let segment = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if segment < output.len() {
        let begin = usize::cast_from(starts[address(starts, segment)]);
        let end = usize::cast_from(ends[address(ends, segment)]);
        let mut re = A::cast_from(0u32); let mut im = A::cast_from(0u32);
        if comptime!(product) { re = A::cast_from(1u32); }
        for position in begin..end {
            let mut entry = position;
            if comptime!(indirect) { entry = usize::cast_from(order[address(order, position)]); }
            let value = A::cast_from(values[address(values, entry)]); let mut value_i = A::cast_from(0u32);
            if comptime!(complex) { value_i = A::cast_from(imaginary[address(imaginary, entry)]); }
            let (next_re, next_im) = combine::<S, A>(re, im, value, value_i, product, complex);
            re = next_re; im = next_im;
        }
        output[address(output, segment)] = O::cast_from(re);
        if comptime!(complex) { output_imaginary[address(output_imaginary, segment)] = O::cast_from(im); }
    }
}

#[ruda(launch)]
fn chunks<S: Numeric + RudaElement, A: Numeric + RudaElement, O: Numeric + RudaElement>(
    values: &Tensor<S>, imaginary: &Tensor<S>, partials: &mut Tensor<A>,
    output: &mut Tensor<O>, output_imaginary: &mut Tensor<O>, count: usize,
    #[comptime] product: bool, #[comptime] complex: bool, #[comptime] direct: bool,
) {
    let chunk = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize; let tid = UNIT_POS as usize;
    if chunk < count {
        let begin = chunk * 65536usize; let remaining = values.len() - begin;
        let mut re = A::cast_from(0u32); let mut im = A::cast_from(0u32); let mut item = tid;
        if comptime!(product) { re = A::cast_from(1u32); }
        while item < remaining && item < 65536usize {
            let value = A::cast_from(values[address(values, begin + item)]); let mut value_i = A::cast_from(0u32);
            if comptime!(complex) { value_i = A::cast_from(imaginary[address(imaginary, begin + item)]); }
            let (next_re, next_im) = combine::<S, A>(re, im, value, value_i, product, complex);
            re = next_re; im = next_im; item += 256usize;
        }
        let (value, value_i) = block_reduce::<S, A>(re, im, product, complex);
        if tid == 0usize {
            if comptime!(direct) {
                output[0] = O::cast_from(S::cast_from(value));
                if comptime!(complex) { output_imaginary[0] = O::cast_from(S::cast_from(value_i)); }
            } else {
                partials[chunk] = value;
                if comptime!(complex) { partials[count + chunk] = value_i; }
            }
        }
    }
}

#[ruda(launch)]
fn finish<S: Numeric, A: Numeric + RudaElement, O: Numeric + RudaElement>(
    partials: &Tensor<A>, output: &mut Tensor<O>, imaginary: &mut Tensor<O>, count: usize,
    #[comptime] product: bool, #[comptime] complex: bool,
) {
    let tid = UNIT_POS as usize; let mut chunk = tid;
    let mut re = A::cast_from(0u32); let mut im = A::cast_from(0u32);
    if comptime!(product) { re = A::cast_from(1u32); }
    while chunk < count {
        let value = partials[chunk]; let mut value_i = A::cast_from(0u32);
        if comptime!(complex) { value_i = partials[count + chunk]; }
        let (next_re, next_im) = combine::<S, A>(re, im, value, value_i, product, complex);
        re = next_re; im = next_im; chunk += 256usize;
    }
    let (value, value_i) = block_reduce::<S, A>(re, im, product, complex);
    if tid == 0usize {
        output[0] = O::cast_from(S::cast_from(value));
        if comptime!(complex) { imaginary[0] = O::cast_from(S::cast_from(value_i)); }
    }
}

fn part(v: &View, imaginary: bool) -> View {
    if matches!(v.dtype, 10 | 11) { scientific::component(v, imaginary) }
    else { View { handle: v.handle.clone(), shape: v.shape.clone(), strides: v.strides.clone(), len: v.len, dtype: v.dtype } }
}

fn grid(blocks: usize) -> RudaCount {
    assert!(blocks > 0 && blocks <= u32::MAX as usize);
    let x = blocks.min(131072);
    RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1)
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64]) {
    assert!(op == 528 || op == 529); assert_eq!(ints.len(), 1); assert!(matches!(ints[0], 0 | 1));
    let segmented = op == 528; assert_eq!(xs.len(), if segmented { 5 } else { 2 });
    let values = &xs[if segmented { 3 } else { 0 }]; let output = &xs[if segmented { 4 } else { 1 }];
    assert_eq!(values.shape.len(), 1); assert_eq!(output.shape.len(), 1);
    let dtype = values.dtype; assert!(matches!(dtype, 0..=2 | 4..=11));
    assert_eq!(output.dtype, if (4..=8).contains(&dtype) { 4 } else { dtype });
    if segmented {
        for v in &xs[..3] { assert_eq!(v.dtype, 4); assert_eq!(v.shape.len(), 1); }
        assert_eq!(xs[0].len, output.len); assert_eq!(xs[1].len, output.len);
        assert!(xs[2].len == 0 || xs[2].len == values.len);
        if output.len == 0 { return; }
    } else {
        assert!(values.len > 0); assert_eq!(output.len, 1);
    }
    let parts: Vec<View> = [values, output].iter().flat_map(|v| [part(v, false), part(v, true)]).collect();
    for v in &parts { scientific::indexable(v); }
    let c = client(); let block = RudaDim::new_1d(256); let product = ints[0] != 0; let complex = matches!(dtype, 10 | 11);
    if segmented {
        macro_rules! run { ($s:ty, $a:ty, $o:ty) => { unsafe {
            segments::launch::<$s, $a, $o, CudaRuntime>(&c, grid(output.len.div_ceil(256)), block,
                xs[0].arg(), xs[1].arg(), xs[2].arg(), parts[0].arg(), parts[1].arg(), parts[2].arg(), parts[3].arg(),
                xs[2].len != 0, product, complex);
        }}; }
        match dtype {
            0 | 10 => run!(f32, f32, f32), 1 => run!(f16, f32, f16), 2 => run!(bf16, f32, bf16),
            9 | 11 => run!(f64, f64, f64), 4 => run!(i64, u64, i64), 5 => run!(i32, u64, i64),
            6 => run!(i16, u64, i64), 7 => run!(i8, u64, i64), 8 => run!(u8, u64, i64), _ => unreachable!(),
        }
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
    } else {
        let count = values.len.div_ceil(65536);
        let length = if count == 1 { 1 } else { count * if complex { 2 } else { 1 } };
        let accumulator = if (4..=8).contains(&dtype) { 4 } else if matches!(dtype, 9 | 11) { 9 } else { 0 };
        let partials = View::packed(c.empty(length * element_bytes(accumulator)), vec![length], accumulator);
        macro_rules! run { ($s:ty, $a:ty, $o:ty) => { unsafe {
            chunks::launch::<$s, $a, $o, CudaRuntime>(&c, grid(count), block, parts[0].arg(), parts[1].arg(), partials.arg(),
                parts[2].arg(), parts[3].arg(), count, product, complex, count == 1);
            if count > 1 {
                finish::launch::<$s, $a, $o, CudaRuntime>(&c, grid(1), block, partials.arg(), parts[2].arg(), parts[3].arg(), count, product, complex);
            }
        }}; }
        match dtype {
            0 | 10 => run!(f32, f32, f32), 1 => run!(f16, f32, f16), 2 => run!(bf16, f32, bf16),
            9 | 11 => run!(f64, f64, f64), 4 => run!(i64, u64, i64), 5 => run!(i32, u64, i64),
            6 => run!(i16, u64, i64), 7 => run!(i8, u64, i64), 8 => run!(u8, u64, i64), _ => unreachable!(),
        }
        LAUNCHES.fetch_add(if count == 1 { 1 } else { 2 }, Ordering::Relaxed);
    }
    finish_dispatch(&c);
}
