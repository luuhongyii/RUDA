use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn storage<S: Float, F: Float>(value: F, #[comptime] code: u32) -> S {
    let mut result = S::new(0.0);
    if comptime!(code == 1 || code == 2) { result = S::cast_from(f32::cast_from(value)); }
    else { result = S::cast_from(value); }
    result
}

#[ruda]
fn block_sum<T: Numeric>(value: T) -> T {
    let tid = UNIT_POS as usize;
    let mut result = value;
    let mut offset = 16u32;
    while offset > 0u32 {
        result += plane_shuffle_down(result, offset);
        offset /= 2u32;
    }
    let mut shared = SharedMemory::<T>::new(8usize);
    sync_ruda();
    if tid % 32usize == 0 { shared[tid / 32usize] = result; }
    sync_ruda();
    result = T::cast_from(0u32);
    if tid < 8usize { result = shared[tid]; }
    if tid < 32usize {
        offset = 16u32;
        while offset > 0u32 {
            result += plane_shuffle_down(result, offset);
            offset /= 2u32;
        }
    }
    result
}

#[ruda]
fn maximum<F: Float>(a: F, b: F) -> F {
    let mut result = a;
    if b != b || b > a { result = b; }
    result
}

#[ruda]
fn block_max<F: Float>(value: F) -> F {
    let tid = UNIT_POS as usize;
    let mut result = value;
    let mut offset = 16u32;
    while offset > 0u32 {
        result = maximum::<F>(result, plane_shuffle_down(result, offset));
        offset /= 2u32;
    }
    let mut shared = SharedMemory::<F>::new(8usize);
    sync_ruda();
    if tid % 32usize == 0 { shared[tid / 32usize] = result; }
    sync_ruda();
    result = F::new(f32::NEG_INFINITY);
    if tid < 8usize { result = shared[tid]; }
    if tid < 32usize {
        offset = 16u32;
        while offset > 0u32 {
            result = maximum::<F>(result, plane_shuffle_down(result, offset));
            offset /= 2u32;
        }
    }
    result
}

#[ruda(launch)]
fn empty<S: Float + RudaElement>(
    real: &mut Tensor<S>, imag: &mut Tensor<S>, #[comptime] op: u32, #[comptime] complex: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < real.len() {
        let mut x = S::new(0.0);
        let mut y = S::new(0.0);
        if comptime!(op == 458) { x = S::new(f32::NAN); y = x; }
        else if comptime!(op == 459) { x = S::new(f32::NEG_INFINITY); }
        real[address(real, p)] = x;
        if comptime!(complex) { imag[address(imag, p)] = y; }
    }
}

#[ruda]
fn write_nan<S: Float, F: Float>(
    out_re: &mut Tensor<S>, out_im: &mut Tensor<S>, row: usize, real: F, imag: F, count: u32,
    #[comptime] mean: bool, #[comptime] complex: bool, #[comptime] code: u32,
) {
    let mut x = F::cast_from(storage::<S, F>(real, code));
    let mut y = F::cast_from(storage::<S, F>(imag, code));
    if comptime!(mean) {
        let divisor = F::cast_from(storage::<S, F>(F::cast_from(count), code));
        if comptime!(complex) {
            let (a, b) = crate::foreach_unary::complex_divide::<F>(x, y, divisor, F::new(0.0));
            x = a; y = b;
        } else { x /= divisor; }
    }
    out_re[address(out_re, row)] = storage::<S, F>(x, code);
    if comptime!(complex) { out_im[address(out_im, row)] = storage::<S, F>(y, code); }
}

#[ruda(launch)]
fn nan_chunks<S: Float + RudaElement, R: Float + RudaElement, F: Float + RudaElement>(
    real: &Tensor<S>, imag: &Tensor<S>, original_re: &Tensor<R>, original_im: &Tensor<R>,
    partial_re: &mut Tensor<F>, partial_im: &mut Tensor<F>, partial_count: &mut Tensor<u32>,
    out_re: &mut Tensor<S>, out_im: &mut Tensor<S>, width: usize, chunks: usize,
    #[comptime] mean: bool, #[comptime] complex: bool, #[comptime] original_complex: bool,
    #[comptime] code: u32, #[comptime] last: bool,
) {
    let task = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    let tid = UNIT_POS as usize;
    if task < out_re.len() * chunks {
        let row = task / chunks;
        let start = task % chunks * 65536usize;
        let remaining = width - start;
        let mut offset = tid;
        let mut sum_re = F::new(0.0);
        let mut sum_im = F::new(0.0);
        let mut count = 0u32;
        while offset < remaining && offset < 65536usize {
            let p = row * width + start + offset;
            let x = F::cast_from(real[address(real, p)]);
            let mut y = F::new(0.0);
            if comptime!(complex) { y = F::cast_from(imag[address(imag, p)]); }
            if x == x && y == y { sum_re += x; sum_im += y; }
            if comptime!(mean) {
                let a = original_re[address(original_re, p)];
                let mut valid = a == a;
                if comptime!(original_complex) {
                    let b = original_im[address(original_im, p)];
                    valid = valid && b == b;
                }
                if valid { count += 1u32; }
            }
            offset += 256usize;
        }
        sum_re = block_sum::<F>(sum_re);
        if comptime!(complex) { sum_im = block_sum::<F>(sum_im); }
        if comptime!(mean) { count = block_sum::<u32>(count); }
        if tid == 0 {
            if comptime!(last) { write_nan::<S, F>(out_re, out_im, row, sum_re, sum_im, count, mean, complex, code); }
            else {
                partial_re[task] = sum_re;
                if comptime!(complex) { partial_im[task] = sum_im; }
                if comptime!(mean) { partial_count[task] = count; }
            }
        }
    }
}

#[ruda(launch)]
fn nan_finish<S: Float + RudaElement, F: Float + RudaElement>(
    partial_re: &Tensor<F>, partial_im: &Tensor<F>, partial_count: &Tensor<u32>,
    out_re: &mut Tensor<S>, out_im: &mut Tensor<S>, chunks: usize,
    #[comptime] mean: bool, #[comptime] complex: bool, #[comptime] code: u32,
) {
    let row = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    let tid = UNIT_POS as usize;
    if row < out_re.len() {
        let mut chunk = tid;
        let mut sum_re = F::new(0.0);
        let mut sum_im = F::new(0.0);
        let mut count = 0u32;
        while chunk < chunks {
            let p = row * chunks + chunk;
            sum_re += partial_re[p];
            if comptime!(complex) { sum_im += partial_im[p]; }
            if comptime!(mean) { count += partial_count[p]; }
            chunk += 256usize;
        }
        sum_re = block_sum::<F>(sum_re);
        if comptime!(complex) { sum_im = block_sum::<F>(sum_im); }
        if comptime!(mean) { count = block_sum::<u32>(count); }
        if tid == 0 { write_nan::<S, F>(out_re, out_im, row, sum_re, sum_im, count, mean, complex, code); }
    }
}

#[ruda(launch)]
fn max_chunks<S: Float + RudaElement, F: Float>(
    real: &Tensor<S>, partials: &mut Tensor<S>, maxima: &mut Tensor<S>, width: usize, chunks: usize,
    #[comptime] last: bool,
) {
    let task = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    let tid = UNIT_POS as usize;
    if task < maxima.len() * chunks {
        let row = task / chunks;
        let start = task % chunks * 65536usize;
        let remaining = width - start;
        let mut offset = tid;
        let mut value = F::new(f32::NEG_INFINITY);
        while offset < remaining && offset < 65536usize {
            value = maximum::<F>(value, F::cast_from(real[address(real, row * width + start + offset)]));
            offset += 256usize;
        }
        value = block_max::<F>(value);
        if tid == 0 {
            if comptime!(last) {
                if value.abs() == F::new(f32::INFINITY) { value = F::new(0.0); }
                maxima[row] = S::cast_from(value);
            } else { partials[task] = S::cast_from(value); }
        }
    }
}

#[ruda(launch)]
fn max_finish<S: Float + RudaElement, F: Float>(partials: &Tensor<S>, maxima: &mut Tensor<S>, chunks: usize) {
    let row = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    let tid = UNIT_POS as usize;
    if row < maxima.len() {
        let mut chunk = tid;
        let mut value = F::new(f32::NEG_INFINITY);
        while chunk < chunks {
            value = maximum::<F>(value, F::cast_from(partials[row * chunks + chunk]));
            chunk += 256usize;
        }
        value = block_max::<F>(value);
        if tid == 0 {
            if value.abs() == F::new(f32::INFINITY) { value = F::new(0.0); }
            maxima[row] = S::cast_from(value);
        }
    }
}

#[ruda]
fn write_log<S: Float, O: Float, F: Float, A: Float>(
    maxima: &Tensor<S>, out_re: &mut Tensor<O>, out_im: &mut Tensor<O>, row: usize, real: F, imag: F,
    #[comptime] complex: bool, #[comptime] output_code: u32,
) {
    let x = F::cast_from(storage::<O, F>(real, output_code));
    let y = F::cast_from(storage::<O, F>(imag, output_code));
    let mut log_re = x.ln();
    let mut log_im = F::new(0.0);
    if comptime!(complex) {
        let (a, b) = crate::scientific::complex_log::<F>(x, y);
        log_re = a; log_im = b;
    }
    let result = A::cast_from(storage::<O, F>(log_re, output_code)) + A::cast_from(maxima[row]);
    out_re[address(out_re, row)] = storage::<O, A>(result, output_code);
    if comptime!(complex) {
        let phase = A::cast_from(storage::<O, F>(log_im, output_code)) + A::new(0.0);
        out_im[address(out_im, row)] = storage::<O, A>(phase, output_code);
    }
}

#[ruda(launch)]
fn exp_chunks<S: Float + RudaElement, O: Float + RudaElement, I: Float, F: Float + RudaElement, A: Float>(
    real: &Tensor<S>, imag: &Tensor<S>, maxima: &Tensor<S>, partial_re: &mut Tensor<F>, partial_im: &mut Tensor<F>,
    out_re: &mut Tensor<O>, out_im: &mut Tensor<O>, width: usize, chunks: usize,
    #[comptime] input_complex: bool, #[comptime] output_complex: bool, #[comptime] input_code: u32,
    #[comptime] output_code: u32, #[comptime] math_code: u32, #[comptime] last: bool,
) {
    let task = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    let tid = UNIT_POS as usize;
    if task < out_re.len() * chunks {
        let row = task / chunks;
        let start = task % chunks * 65536usize;
        let remaining = width - start;
        let shift = I::cast_from(maxima[row]);
        let mut offset = tid;
        let mut sum_re = F::new(0.0);
        let mut sum_im = F::new(0.0);
        while offset < remaining && offset < 65536usize {
            let p = row * width + start + offset;
            let x = I::cast_from(storage::<S, I>(I::cast_from(real[address(real, p)]) - shift, input_code));
            let mut exp_re = x.exp();
            let mut exp_im = I::new(0.0);
            if comptime!(input_complex) {
                let y = I::cast_from(imag[address(imag, p)]);
                let (a, b) = crate::foreach_unary::complex_exp::<I>(x, y, math_code);
                exp_re = a; exp_im = b;
            }
            let stored_re = storage::<S, I>(exp_re, input_code);
            sum_re += F::cast_from(storage::<O, S>(stored_re, output_code));
            if comptime!(output_complex) {
                let stored_im = storage::<S, I>(exp_im, input_code);
                sum_im += F::cast_from(storage::<O, S>(stored_im, output_code));
            }
            offset += 256usize;
        }
        sum_re = block_sum::<F>(sum_re);
        if comptime!(output_complex) { sum_im = block_sum::<F>(sum_im); }
        if tid == 0 {
            if comptime!(last) {
                write_log::<S, O, F, A>(maxima, out_re, out_im, row, sum_re, sum_im, output_complex, output_code);
            } else {
                partial_re[task] = sum_re;
                if comptime!(output_complex) { partial_im[task] = sum_im; }
            }
        }
    }
}

#[ruda(launch)]
fn log_finish<S: Float + RudaElement, O: Float + RudaElement, F: Float + RudaElement, A: Float>(
    maxima: &Tensor<S>, partial_re: &Tensor<F>, partial_im: &Tensor<F>, out_re: &mut Tensor<O>, out_im: &mut Tensor<O>,
    chunks: usize, #[comptime] complex: bool, #[comptime] output_code: u32,
) {
    let row = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    let tid = UNIT_POS as usize;
    if row < out_re.len() {
        let mut chunk = tid;
        let mut sum_re = F::new(0.0);
        let mut sum_im = F::new(0.0);
        while chunk < chunks {
            let p = row * chunks + chunk;
            sum_re += partial_re[p];
            if comptime!(complex) { sum_im += partial_im[p]; }
            chunk += 256usize;
        }
        sum_re = block_sum::<F>(sum_re);
        if comptime!(complex) { sum_im = block_sum::<F>(sum_im); }
        if tid == 0 { write_log::<S, O, F, A>(maxima, out_re, out_im, row, sum_re, sum_im, complex, output_code); }
    }
}

fn grid(tasks: usize) -> RudaCount {
    assert!(tasks > 0 && tasks <= u32::MAX as usize, "NaN/log reduction launch exceeds U32 indexing");
    let x = tasks.min(131072);
    RudaCount::Static(x as u32, tasks.div_ceil(x) as u32, 1)
}

fn scratch(c: &ComputeClient<CudaRuntime>, len: usize, code: u32) -> View {
    View::packed(c.empty(len.max(1).checked_mul(element_bytes(code)).expect("NaN/log reduction workspace overflow")), vec![len], code)
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64]) {
    assert!(matches!(op, 457..=459));
    assert_eq!(xs.len(), 6);
    assert_eq!(ints.len(), 3);
    let real = &xs[0]; let imag = &xs[1];
    let original_re = &xs[2]; let original_im = &xs[3];
    let out_re = &xs[4]; let out_im = &xs[5];
    assert!(ints[0] >= 0 && ints[0] as usize <= real.shape.len());
    assert!(matches!(ints[1], 0 | 1) && matches!(ints[2], 0 | 1));
    for pair in [&xs[..2], &xs[2..4], &xs[4..]] {
        assert!(matches!(pair[0].dtype, 0 | 1 | 2 | 9));
        assert_eq!(pair[0].dtype, pair[1].dtype); assert_eq!(pair[0].shape, pair[1].shape);
    }
    assert_eq!(real.shape, original_re.shape);
    let outer_rank = ints[0] as usize;
    let product = |shape: &[usize]| shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)).expect("NaN/log reduction shape overflow");
    let rows = product(&real.shape[..outer_rank]);
    let width = product(&real.shape[outer_rank..]);
    assert_eq!(out_re.shape, vec![rows]);
    assert_eq!(real.len, rows.checked_mul(width).expect("NaN/log reduction element count overflow"));
    if rows == 0 { return; }
    let input_complex = ints[1] != 0;
    let output_complex = if op == 459 { ints[2] != 0 } else { input_complex };
    let c = client();
    let block = RudaDim::new_1d(256);
    if width == 0 {
        macro_rules! fill { ($s:ty) => { unsafe { empty::launch::<$s, CudaRuntime>(&c, grid(rows.div_ceil(256)), block,
            out_re.arg(), out_im.arg(), op, output_complex); }}; }
        match out_re.dtype { 0 => fill!(f32), 1 => fill!(f16), 2 => fill!(bf16), 9 => fill!(f64), _ => unreachable!() }
        LAUNCHES.fetch_add(1, Ordering::Relaxed); finish_dispatch(&c); return;
    }
    let chunks = width.div_ceil(65536);
    let tasks = rows.checked_mul(chunks).filter(|&n| n <= u32::MAX as usize).expect("NaN/log reduction partial count overflow");
    let code = if out_re.dtype == 9 { 9 } else { 0 };
    let partial_len = if chunks > 1 { tasks } else { 1 };
    let partial_re = scratch(&c, partial_len, code);
    let partial_im = scratch(&c, if output_complex { partial_len } else { 1 }, code);
    if op != 459 {
        assert_eq!(out_re.dtype, real.dtype);
        let mean = op == 458;
        let counts = scratch(&c, if mean { partial_len } else { 1 }, 5);
        macro_rules! run { ($s:ty, $r:ty, $f:ty) => { unsafe {
            nan_chunks::launch::<$s, $r, $f, CudaRuntime>(&c, grid(tasks), block,
                real.arg(), imag.arg(), original_re.arg(), original_im.arg(), partial_re.arg(), partial_im.arg(), counts.arg(),
                out_re.arg(), out_im.arg(), width, chunks, mean, input_complex, ints[2] != 0, real.dtype, chunks == 1);
            if chunks > 1 { nan_finish::launch::<$s, $f, CudaRuntime>(&c, grid(rows), block,
                partial_re.arg(), partial_im.arg(), counts.arg(), out_re.arg(), out_im.arg(), chunks, mean, input_complex, real.dtype); }
        }}; }
        macro_rules! original { ($s:ty, $f:ty) => { match original_re.dtype {
            0 => run!($s, f32, $f), 1 => run!($s, f16, $f), 2 => run!($s, bf16, $f), 9 => run!($s, f64, $f), _ => unreachable!(),
        }}; }
        match real.dtype { 0 => original!(f32, f32), 1 => original!(f16, f32), 2 => original!(bf16, f32), 9 => original!(f64, f64), _ => unreachable!() }
        LAUNCHES.fetch_add(if chunks > 1 { 2 } else { 1 }, Ordering::Relaxed);
    } else {
        let maxima = scratch(&c, rows, real.dtype);
        let max_partials = scratch(&c, partial_len, real.dtype);
        macro_rules! max { ($s:ty, $f:ty) => { unsafe {
            max_chunks::launch::<$s, $f, CudaRuntime>(&c, grid(tasks), block, real.arg(), max_partials.arg(), maxima.arg(), width, chunks, chunks == 1);
            if chunks > 1 { max_finish::launch::<$s, $f, CudaRuntime>(&c, grid(rows), block, max_partials.arg(), maxima.arg(), chunks); }
        }}; }
        match real.dtype { 0 => max!(f32, f32), 1 => max!(f16, f32), 2 => max!(bf16, f32), 9 => max!(f64, f64), _ => unreachable!() }
        macro_rules! run { ($s:ty, $o:ty, $i:ty, $f:ty, $a:ty) => { unsafe {
            exp_chunks::launch::<$s, $o, $i, $f, $a, CudaRuntime>(&c, grid(tasks), block,
                real.arg(), imag.arg(), maxima.arg(), partial_re.arg(), partial_im.arg(), out_re.arg(), out_im.arg(), width, chunks,
                input_complex, output_complex, real.dtype, out_re.dtype, if real.dtype == 9 { 9 } else { 0 }, chunks == 1);
            if chunks > 1 { log_finish::launch::<$s, $o, $f, $a, CudaRuntime>(&c, grid(rows), block,
                maxima.arg(), partial_re.arg(), partial_im.arg(), out_re.arg(), out_im.arg(), chunks, output_complex, out_re.dtype); }
        }}; }
        macro_rules! output { ($s:ty, $i:ty) => { match out_re.dtype {
            0 => run!($s, f32, $i, f32, $i), 1 => run!($s, f16, $i, f32, $i),
            2 => run!($s, bf16, $i, f32, $i), 9 => run!($s, f64, $i, f64, f64), _ => unreachable!(),
        }}; }
        match real.dtype { 0 => output!(f32, f32), 1 => output!(f16, f32), 2 => output!(bf16, f32), 9 => output!(f64, f64), _ => unreachable!() }
        LAUNCHES.fetch_add(if chunks > 1 { 4 } else { 2 }, Ordering::Relaxed);
    }
    finish_dispatch(&c);
}
