use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::{Tensor, fma};

#[ruda]
fn storage<O: Float, F: Float>(value: F, #[comptime] code: u32) -> O {
    let mut result = O::new(0.0);
    if comptime!(code == 1 || code == 2) { result = O::cast_from(f32::cast_from(value)); }
    else { result = O::cast_from(value); }
    result
}

#[ruda]
fn update<F: Float>(mean: F, moment: F, count: u32, value: F) -> (F, F) {
    let delta = value - mean;
    let next_mean = mean + delta / F::cast_from(count);
    let next_delta = value - next_mean;
    (next_mean, fma(delta, next_delta, moment))
}

#[ruda]
fn combine<F: Float>(am: F, av: F, an: F, bm: F, bv: F, bn: F) -> (F, F, F) {
    let mut mean = am;
    let mut moment = av;
    let mut count = an;
    if an == F::new(0.0) { mean = bm; moment = bv; count = bn; }
    else if bn != F::new(0.0) {
        let delta = bm - am;
        count = an + bn;
        let weight = bn / count;
        mean = fma(delta, weight, am);
        moment = fma(delta * delta * an, weight, av + bv);
    }
    (mean, moment, count)
}

#[ruda]
fn block_moments<F: Float>(mean: F, moment: F, count: F) -> (F, F, F) {
    let tid = UNIT_POS as usize;
    let mut m = mean;
    let mut v = moment;
    let mut n = count;
    let mut offset = 16u32;
    while offset > 0u32 {
        let (next_m, next_v, next_n) = combine(m, v, n,
            plane_shuffle_down(m, offset), plane_shuffle_down(v, offset), plane_shuffle_down(n, offset));
        m = next_m; v = next_v; n = next_n;
        offset /= 2u32;
    }
    let mut shared_m = SharedMemory::<F>::new(8usize);
    let mut shared_v = SharedMemory::<F>::new(8usize);
    let mut shared_n = SharedMemory::<F>::new(8usize);
    sync_ruda();
    if tid % 32usize == 0 {
        shared_m[tid / 32usize] = m;
        shared_v[tid / 32usize] = v;
        shared_n[tid / 32usize] = n;
    }
    sync_ruda();
    m = F::new(0.0); v = F::new(0.0); n = F::new(0.0);
    if tid < 8usize { m = shared_m[tid]; v = shared_v[tid]; n = shared_n[tid]; }
    if tid < 32usize {
        offset = 16u32;
        while offset > 0u32 {
            let (next_m, next_v, next_n) = combine(m, v, n,
                plane_shuffle_down(m, offset), plane_shuffle_down(v, offset), plane_shuffle_down(n, offset));
            m = next_m; v = next_v; n = next_n;
            offset /= 2u32;
        }
    }
    (m, v, n)
}

#[ruda]
fn write_result<S: Float, O: Float, F: Float>(
    variance_re: &mut Tensor<O>, variance_im: &mut Tensor<O>, mean_re: &mut Tensor<O>, mean_im: &mut Tensor<O>,
    row: usize, mr: F, vr: F, mi: F, vi: F, count: F, correction: F,
    #[comptime] complex: bool, #[comptime] std: bool, #[comptime] with_mean: bool,
    #[comptime] variance_complex: bool, #[comptime] rounded_mean: bool, #[comptime] output_code: u32,
) {
    let mut divisor = F::new(0.0);
    if count > correction { divisor = count - correction; }
    let mut variance = vr / divisor;
    let mut imaginary = F::new(0.0);
    if comptime!(complex) {
        let real_variance = F::cast_from(storage::<O, F>(variance, output_code));
        let imag_variance = F::cast_from(storage::<O, F>(vi / divisor, output_code));
        variance = F::cast_from(storage::<O, F>(real_variance + imag_variance, output_code));
    }
    if comptime!(std) {
        if comptime!(variance_complex) {
            let (real, imag) = crate::scientific::complex_sqrt(variance, F::new(0.0));
            variance = real; imaginary = imag;
        } else { variance = variance.sqrt(); }
    }
    variance_re[address(variance_re, row)] = storage::<O, F>(variance, output_code);
    if comptime!(variance_complex) { variance_im[address(variance_im, row)] = storage::<O, F>(imaginary, output_code); }
    if comptime!(with_mean) {
        let mut real_mean = mr;
        let mut imag_mean = mi;
        if comptime!(rounded_mean) {
            real_mean = F::cast_from(S::cast_from(real_mean));
            imag_mean = F::cast_from(S::cast_from(imag_mean));
        }
        mean_re[address(mean_re, row)] = storage::<O, F>(real_mean, output_code);
        if comptime!(complex) { mean_im[address(mean_im, row)] = storage::<O, F>(imag_mean, output_code); }
    }
}

#[ruda(launch)]
fn empty<O: Float + RudaElement>(
    variance_re: &mut Tensor<O>, variance_im: &mut Tensor<O>, mean_re: &mut Tensor<O>, mean_im: &mut Tensor<O>,
    #[comptime] complex: bool, #[comptime] std: bool, #[comptime] with_mean: bool, #[comptime] variance_complex: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < variance_re.len() {
        variance_re[address(variance_re, p)] = O::new(f32::NAN);
        if comptime!(variance_complex) {
            let mut value = O::new(0.0);
            if comptime!(complex && std) { value = O::new(f32::NAN); }
            variance_im[address(variance_im, p)] = value;
        }
        if comptime!(with_mean) {
            mean_re[address(mean_re, p)] = O::new(f32::NAN);
            if comptime!(complex) { mean_im[address(mean_im, p)] = O::new(f32::NAN); }
        }
    }
}

#[ruda(launch)]
fn chunks<S: Float + RudaElement, O: Float + RudaElement, F: Float + RudaElement>(
    real: &Tensor<S>, imag: &Tensor<S>, partials: &mut Tensor<F>,
    variance_re: &mut Tensor<O>, variance_im: &mut Tensor<O>, mean_re: &mut Tensor<O>, mean_im: &mut Tensor<O>,
    width: usize, count: usize, correction: F,
    #[comptime] complex: bool, #[comptime] std: bool, #[comptime] with_mean: bool,
    #[comptime] variance_complex: bool, #[comptime] rounded_mean: bool, #[comptime] output_code: u32, #[comptime] last: bool,
) {
    let task = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    let tid = UNIT_POS as usize;
    let tasks = variance_re.len() * count;
    if task < tasks {
        let row = task / count;
        let start = task % count * 65536usize;
        let remaining = width - start;
        let mut offset = tid;
        let mut mr = F::new(0.0); let mut vr = F::new(0.0);
        let mut mi = F::new(0.0); let mut vi = F::new(0.0);
        let mut n = 0u32;
        while offset < remaining && offset < 65536usize {
            let p = row * width + start + offset;
            let x = F::cast_from(storage::<O, S>(real[address(real, p)], output_code));
            n += 1u32;
            let (next_mean, next_moment) = update(mr, vr, n, x);
            mr = next_mean; vr = next_moment;
            if comptime!(complex) {
                let y = F::cast_from(storage::<O, S>(imag[address(imag, p)], output_code));
                let (next_mean, next_moment) = update(mi, vi, n, y);
                mi = next_mean; vi = next_moment;
            }
            offset += 256usize;
        }
        let nf = F::cast_from(n);
        let (real_mean, real_moment, total) = block_moments(mr, vr, nf);
        mr = real_mean; vr = real_moment;
        if comptime!(complex) {
            let (imag_mean, imag_moment, _) = block_moments(mi, vi, nf);
            mi = imag_mean; vi = imag_moment;
        }
        if tid == 0 {
            if comptime!(last) {
                write_result::<S, O, F>(variance_re, variance_im, mean_re, mean_im, row, mr, vr, mi, vi, total, correction,
                    complex, std, with_mean, variance_complex, rounded_mean, output_code);
            } else {
                partials[task] = mr; partials[tasks + task] = vr; partials[2usize * tasks + task] = total;
                if comptime!(complex) { partials[3usize * tasks + task] = mi; partials[4usize * tasks + task] = vi; }
            }
        }
    }
}

#[ruda(launch)]
fn finish<S: Float + RudaElement, O: Float + RudaElement, F: Float + RudaElement>(
    partials: &Tensor<F>, variance_re: &mut Tensor<O>, variance_im: &mut Tensor<O>,
    mean_re: &mut Tensor<O>, mean_im: &mut Tensor<O>, count: usize, correction: F,
    #[comptime] complex: bool, #[comptime] std: bool, #[comptime] with_mean: bool,
    #[comptime] variance_complex: bool, #[comptime] rounded_mean: bool, #[comptime] output_code: u32,
) {
    let row = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    let tid = UNIT_POS as usize;
    let tasks = variance_re.len() * count;
    if row < variance_re.len() {
        let mut chunk = tid;
        let mut mr = F::new(0.0); let mut vr = F::new(0.0);
        let mut mi = F::new(0.0); let mut vi = F::new(0.0);
        let mut n = F::new(0.0);
        while chunk < count {
            let p = row * count + chunk;
            let next_count = partials[2usize * tasks + p];
            let (real_mean, real_moment, total) = combine(mr, vr, n, partials[p], partials[tasks + p], next_count);
            mr = real_mean; vr = real_moment;
            if comptime!(complex) {
                let (imag_mean, imag_moment, _) = combine(mi, vi, n, partials[3usize * tasks + p], partials[4usize * tasks + p], next_count);
                mi = imag_mean; vi = imag_moment;
            }
            n = total;
            chunk += 256usize;
        }
        let (real_mean, real_moment, total) = block_moments(mr, vr, n);
        mr = real_mean; vr = real_moment;
        if comptime!(complex) {
            let (imag_mean, imag_moment, _) = block_moments(mi, vi, n);
            mi = imag_mean; vi = imag_moment;
        }
        if tid == 0 { write_result::<S, O, F>(variance_re, variance_im, mean_re, mean_im, row, mr, vr, mi, vi, total, correction,
            complex, std, with_mean, variance_complex, rounded_mean, output_code); }
    }
}

fn grid(tasks: usize) -> RudaCount {
    assert!(tasks > 0 && tasks <= u32::MAX as usize, "variance launch exceeds U32 indexing");
    let x = tasks.min(131072);
    RudaCount::Static(x as u32, tasks.div_ceil(x) as u32, 1)
}

pub(super) fn dispatch(xs: &[View], ints: &[i64], values: &[f64]) {
    assert_eq!(xs.len(), 6); assert_eq!(ints.len(), 5); assert_eq!(values.len(), 1);
    let real = &xs[0]; let imag = &xs[1];
    let variance_re = &xs[2]; let variance_im = &xs[3]; let mean_re = &xs[4]; let mean_im = &xs[5];
    assert!(ints[0] >= 0 && ints[0] as usize <= real.shape.len());
    assert!(ints[1..].iter().all(|&v| matches!(v, 0 | 1)));
    assert!(matches!(real.dtype, 0 | 1 | 2 | 9) && matches!(variance_re.dtype, 0 | 1 | 2 | 9));
    assert_eq!(imag.dtype, real.dtype); assert_eq!(imag.shape, real.shape);
    let outer_rank = ints[0] as usize;
    let complex = ints[1] != 0; let std = ints[2] != 0; let with_mean = ints[3] != 0; let variance_complex = ints[4] != 0;
    let product = |shape: &[usize]| shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)).expect("variance shape overflow");
    let rows = product(&real.shape[..outer_rank]); let width = product(&real.shape[outer_rank..]);
    assert_eq!(real.len, rows.checked_mul(width).expect("variance element count overflow"));
    for out in &xs[2..] { assert_eq!(out.shape, vec![rows]); assert_eq!(out.dtype, variance_re.dtype); }
    if with_mean { assert!(!variance_complex); }
    if rows == 0 { return; }
    let c = client(); let block = RudaDim::new_1d(256);
    if width == 0 {
        macro_rules! fill { ($o:ty) => { unsafe { empty::launch::<$o, CudaRuntime>(&c, grid(rows.div_ceil(256)), block,
            variance_re.arg(), variance_im.arg(), mean_re.arg(), mean_im.arg(), complex, std, with_mean, variance_complex); }}; }
        match variance_re.dtype { 0 => fill!(f32), 1 => fill!(f16), 2 => fill!(bf16), 9 => fill!(f64), _ => unreachable!() }
        LAUNCHES.fetch_add(1, Ordering::Relaxed); finish_dispatch(&c); return;
    }
    assert!(complex || !variance_complex, "real variance reduction requires real output");
    let count = width.div_ceil(65536);
    let tasks = rows.checked_mul(count).filter(|&n| n <= u32::MAX as usize).expect("variance partial count overflow");
    let code = if variance_re.dtype == 9 { 9 } else { 0 };
    let partial_len = if count > 1 { tasks.checked_mul(if complex { 5 } else { 3 }).expect("variance workspace overflow") } else { 1 };
    assert!(partial_len <= u32::MAX as usize);
    let partials = View::packed(c.empty(partial_len.checked_mul(element_bytes(code)).expect("variance workspace bytes overflow")), vec![partial_len], code);
    let rounded_mean = real.dtype == 1 && variance_re.dtype == 0;
    macro_rules! run { ($s:ty, $o:ty, $f:ty) => { unsafe {
        chunks::launch::<$s, $o, $f, CudaRuntime>(&c, grid(tasks), block, real.arg(), imag.arg(), partials.arg(),
            variance_re.arg(), variance_im.arg(), mean_re.arg(), mean_im.arg(), width, count, values[0] as $f,
            complex, std, with_mean, variance_complex, rounded_mean, variance_re.dtype, count == 1);
        if count > 1 { finish::launch::<$s, $o, $f, CudaRuntime>(&c, grid(rows), block, partials.arg(),
            variance_re.arg(), variance_im.arg(), mean_re.arg(), mean_im.arg(), count, values[0] as $f,
            complex, std, with_mean, variance_complex, rounded_mean, variance_re.dtype); }
    }}; }
    macro_rules! output { ($s:ty) => { match variance_re.dtype {
        0 => run!($s, f32, f32), 1 => run!($s, f16, f32), 2 => run!($s, bf16, f32), 9 => run!($s, f64, f64), _ => unreachable!(),
    }}; }
    match real.dtype { 0 => output!(f32), 1 => output!(f16), 2 => output!(bf16), 9 => output!(f64), _ => unreachable!() }
    LAUNCHES.fetch_add(if count > 1 { 2 } else { 1 }, Ordering::Relaxed);
    finish_dispatch(&c);
}
