use super::*;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn real_op<F: Float>(a: F, b: F, #[comptime] op: u32) -> F {
    let mut result = F::new(0.0);
    if comptime!(op == 0) { result = a + b; }
    else if comptime!(op == 1) { result = a - b; }
    else if comptime!(op == 2) { result = a * b; }
    else if comptime!(op == 3) { result = a / b; }
    else if comptime!(op == 4) {
        result = b;
        if a != a || a > b { result = a; }
    } else if comptime!(op == 5) {
        result = b;
        if a != a || a < b { result = a; }
    } else if comptime!(op == 6) { result = a.powf(b); }
    result
}

#[ruda]
fn integer_power<S: Numeric>(a: S, b: S, #[comptime] signed: bool) -> S {
    let zero = S::cast_from(0u32);
    let one = S::cast_from(1u32);
    let two = S::cast_from(2u32);
    let mut result = one;
    let mut base = a;
    let mut exponent = b;
    let mut negative = false;
    if comptime!(signed) { negative = b < zero; }
    if negative {
        result = zero;
        if a == one { result = one; }
        else if a == zero - one {
            result = one;
            if b % two != zero { result = zero - one; }
        }
    } else {
        while exponent != zero {
            if exponent % two != zero { result *= base; }
            exponent /= two;
            base *= base;
        }
    }
    result
}

#[ruda(launch)]
fn floating<S: Float + RudaElement, F: Float>(
    inputs: &Sequence<Tensor<S>>, rhs: &Sequence<Tensor<S>>, outputs: &mut Sequence<Tensor<S>>,
    scalars: &Sequence<u64>, alpha: u64,
    #[comptime] op: u32, #[comptime] mode: u32, #[comptime] reverse: bool, #[comptime] code: u32,
) {
    let scale = crate::floating::from_bits::<F>(alpha, code);
    let block = RUDA_POS_X as usize;
    let mut first = 0usize;
    #[unroll]
    for tensor in 0..inputs.len() {
        let input = inputs.index(tensor);
        let blocks = input.len() / 65536usize + usize::cast_from(input.len() % 65536usize != 0);
        if block >= first && block < first + blocks {
            let other = rhs.index(tensor);
            let output = outputs.index_mut(tensor);
            let scalar = crate::floating::from_bits::<F>(*scalars.index(tensor), code);
            let base = (block - first) * 65536usize;
            let remaining = input.len() - base;
            let mut offset = UNIT_POS as usize;
            while offset < remaining && offset < 65536usize {
                let p = base + offset;
                let a = F::cast_from(input[p]);
                let mut b = scalar;
                if comptime!(mode == 0) { b = scale * F::cast_from(other[p]); }
                else if comptime!(mode == 2) { b = scale * F::cast_from(other[0usize]); }
                let mut value = F::new(0.0);
                if comptime!(reverse) { value = real_op(b, a, op); }
                else { value = real_op(a, b, op); }
                output[p] = S::cast_from(value);
                offset += 512usize;
            }
        }
        first += blocks;
    }
}

#[ruda(launch)]
fn integral<S: Numeric + RudaElement>(
    inputs: &Sequence<Tensor<S>>, rhs: &Sequence<Tensor<S>>, outputs: &mut Sequence<Tensor<S>>,
    scalars: &Sequence<u64>, alpha: u64,
    #[comptime] op: u32, #[comptime] mode: u32, #[comptime] code: u32,
) {
    let zero = S::cast_from(0u32);
    let scale = S::cast_from(alpha);
    let block = RUDA_POS_X as usize;
    let mut first = 0usize;
    #[unroll]
    for tensor in 0..inputs.len() {
        let input = inputs.index(tensor);
        let blocks = input.len() / 65536usize + usize::cast_from(input.len() % 65536usize != 0);
        if block >= first && block < first + blocks {
            let other = rhs.index(tensor);
            let output = outputs.index_mut(tensor);
            let scalar = S::cast_from(*scalars.index(tensor));
            let base = (block - first) * 65536usize;
            let remaining = input.len() - base;
            let mut offset = UNIT_POS as usize;
            while offset < remaining && offset < 65536usize {
                let p = base + offset;
                let a = input[p];
                let mut b = scalar;
                if comptime!(mode == 0) { b = other[p]; }
                else if comptime!(mode == 2) { b = other[0usize]; }
                let mut value = zero;
                if comptime!(code == 3) {
                    let left = a != zero;
                    let mut right = b != zero;
                    if comptime!(mode != 1) { right = right && scale != zero; }
                    if comptime!(op == 0) { value = S::cast_from(left || right); }
                    else if comptime!(op == 1) { value = S::cast_from(left != right); }
                    else if comptime!(op == 2) { value = S::cast_from(left && right); }
                } else {
                    if comptime!(mode != 1) { b = scale * b; }
                    if comptime!(op == 0) { value = a + b; }
                    else if comptime!(op == 1) { value = a - b; }
                    else if comptime!(op == 2) { value = a * b; }
                    else if comptime!(op == 6) { value = integer_power(b, a, code != 8); }
                }
                output[p] = value;
                offset += 512usize;
            }
        }
        first += blocks;
    }
}

#[ruda]
fn complex_op<F: Float>(a: F, ai: F, b: F, bi: F, #[comptime] op: u32, #[comptime] code: u32) -> (F, F) {
    let mut real = F::new(0.0);
    let mut imag = F::new(0.0);
    if comptime!(op == 0) { real = a + b; imag = ai + bi; }
    else if comptime!(op == 1) { real = a - b; imag = ai - bi; }
    else if comptime!(op == 2) { real = a * b - ai * bi; imag = a * bi + ai * b; }
    else if comptime!(op == 3) {
        let (r, i) = crate::foreach_unary::complex_divide(a, ai, b, bi);
        real = r; imag = i;
    } else if comptime!(op == 6) {
        let (r, i) = crate::scientific::complex_log(a, ai);
        let (pr, pi) = crate::foreach_unary::complex_exp(r * b - i * bi, r * bi + i * b, code);
        real = pr; imag = pi;
    }
    (real, imag)
}

#[ruda(launch)]
fn complex<F: Float + RudaElement>(
    inputs: &Sequence<Tensor<F>>, rhs: &Sequence<Tensor<F>>, outputs: &mut Sequence<Tensor<F>>,
    scalars: &Sequence<u64>, imag_scalars: &Sequence<u64>, alpha: u64, imag_alpha: u64,
    #[comptime] op: u32, #[comptime] mode: u32, #[comptime] reverse: bool, #[comptime] code: u32,
) {
    let scale = crate::floating::from_bits::<F>(alpha, code);
    let imag_scale = crate::floating::from_bits::<F>(imag_alpha, code);
    let block = RUDA_POS_X as usize;
    let mut first = 0usize;
    #[unroll]
    for tensor in 0..inputs.len() {
        let input = inputs.index(tensor);
        let len = input.len() / 2usize;
        let blocks = len / 65536usize + usize::cast_from(len % 65536usize != 0);
        if block >= first && block < first + blocks {
            let other = rhs.index(tensor);
            let output = outputs.index_mut(tensor);
            let scalar = crate::floating::from_bits::<F>(*scalars.index(tensor), code);
            let imag_scalar = crate::floating::from_bits::<F>(*imag_scalars.index(tensor), code);
            let base = (block - first) * 65536usize;
            let remaining = len - base;
            let mut offset = UNIT_POS as usize;
            while offset < remaining && offset < 65536usize {
                let p = (base + offset) * 2usize;
                let a = input[p];
                let ai = input[p + 1usize];
                let mut b = scalar;
                let mut bi = imag_scalar;
                if comptime!(mode != 1) {
                    let mut q = p;
                    if comptime!(mode == 2) { q = 0usize; }
                    let br = other[q];
                    let bj = other[q + 1usize];
                    b = scale * br - imag_scale * bj;
                    bi = scale * bj + imag_scale * br;
                }
                let mut real = F::new(0.0);
                let mut imag = F::new(0.0);
                if comptime!(reverse) {
                    let (r, i) = complex_op(b, bi, a, ai, op, code);
                    real = r; imag = i;
                } else {
                    let (r, i) = complex_op(a, ai, b, bi, op, code);
                    real = r; imag = i;
                }
                output[p] = real;
                output[p + 1usize] = imag;
                offset += 512usize;
            }
        }
        first += blocks;
    }
}

pub(super) fn dispatch(xs: &[View], ints: &[i64]) {
    assert!(ints.len() >= 6);
    assert!((0..=6).contains(&ints[0]) && (0..=2).contains(&ints[1]));
    assert!((0..=1).contains(&ints[2]) && ints[3] > 0);
    let op = ints[0] as u32;
    let mode = ints[1] as u32;
    let reverse = ints[2] != 0;
    let n = usize::try_from(ints[3]).expect("foreach binary tensor count overflow");
    assert_eq!(xs.len(), n.checked_mul(3).expect("foreach binary tensor count overflow"));
    assert_eq!(ints.len(), n.checked_mul(2).and_then(|v| v.checked_add(6)).expect("foreach binary scalar count overflow"));
    let code = xs[0].dtype;
    assert!(code <= 11);
    if reverse { assert!(op == 6 && mode == 1 && code != 3); }
    if matches!(code, 3..=8) { assert!(op <= 2 || reverse); }
    if matches!(code, 10 | 11) { assert!(op != 4 && op != 5); }
    for i in 0..n {
        assert_eq!(xs[i].dtype, code);
        assert_eq!(xs[n + i].dtype, code);
        assert_eq!(xs[2 * n + i].dtype, code);
        assert_eq!(xs[i].shape, vec![xs[i].len]);
        assert_eq!(xs[i].strides, vec![1usize]);
        let rhs_len = if mode == 2 { 1 } else { xs[i].len };
        assert_eq!(xs[n + i].shape, vec![rhs_len]);
        assert_eq!(xs[n + i].strides, vec![1usize]);
        assert_eq!(xs[2 * n + i].shape, xs[i].shape);
        assert_eq!(xs[2 * n + i].strides, vec![1usize]);
    }
    if xs[..n].iter().all(|x| x.len == 0) { return; }
    let storage: Vec<_> = xs.iter().map(|x| {
        let width = if matches!(code, 10 | 11) { 2 } else { 1 };
        let len = x.len.checked_mul(width).expect("foreach binary component count overflow");
        let dtype = match code { 10 => 0, 11 => 9, _ => code };
        let view = View { handle: x.handle.clone(), shape: vec![len], strides: vec![1], len, dtype };
        crate::scientific::indexable(&view);
        view
    }).collect();
    let indices: Vec<_> = (0..n).filter(|&i| xs[i].len != 0).collect();
    let c = client();
    for batch in indices.chunks(16) {
        let blocks: usize = batch.iter().map(|&i| xs[i].len.div_ceil(65536)).sum();
        let grid = RudaCount::Static(u32::try_from(blocks).expect("foreach binary grid overflow"), 1, 1);
        let block = RudaDim::new_1d(512);
        let scalars = batch.iter().map(|&i| ints[6 + i * 2] as u64).collect::<SequenceArg<CudaRuntime, u64>>();
        let alpha = ints[4] as u64;
        macro_rules! floating { ($s:ty, $f:ty, $code:expr) => { unsafe {
            let inputs = batch.iter().map(|&i| storage[i].arg()).collect::<SequenceArg<CudaRuntime, Tensor<$s>>>();
            let rhs = batch.iter().map(|&i| storage[n + i].arg()).collect::<SequenceArg<CudaRuntime, Tensor<$s>>>();
            let outputs = batch.iter().map(|&i| storage[2 * n + i].arg()).collect::<SequenceArg<CudaRuntime, Tensor<$s>>>();
            floating::launch::<$s, $f, CudaRuntime>(&c, grid, block, inputs, rhs, outputs, scalars, alpha, op, mode, reverse, $code);
        }}; }
        macro_rules! integral { ($s:ty) => { unsafe {
            let inputs = batch.iter().map(|&i| storage[i].arg()).collect::<SequenceArg<CudaRuntime, Tensor<$s>>>();
            let rhs = batch.iter().map(|&i| storage[n + i].arg()).collect::<SequenceArg<CudaRuntime, Tensor<$s>>>();
            let outputs = batch.iter().map(|&i| storage[2 * n + i].arg()).collect::<SequenceArg<CudaRuntime, Tensor<$s>>>();
            integral::launch::<$s, CudaRuntime>(&c, grid, block, inputs, rhs, outputs, scalars, alpha, op, mode, code);
        }}; }
        macro_rules! complex { ($f:ty, $code:expr) => { unsafe {
            let inputs = batch.iter().map(|&i| storage[i].arg()).collect::<SequenceArg<CudaRuntime, Tensor<$f>>>();
            let rhs = batch.iter().map(|&i| storage[n + i].arg()).collect::<SequenceArg<CudaRuntime, Tensor<$f>>>();
            let outputs = batch.iter().map(|&i| storage[2 * n + i].arg()).collect::<SequenceArg<CudaRuntime, Tensor<$f>>>();
            let imag_scalars = batch.iter().map(|&i| ints[7 + i * 2] as u64).collect::<SequenceArg<CudaRuntime, u64>>();
            complex::launch::<$f, CudaRuntime>(&c, grid, block, inputs, rhs, outputs, scalars, imag_scalars, alpha, ints[5] as u64, op, mode, reverse, $code);
        }}; }
        match code {
            0 => floating!(f32, f32, 0), 1 => floating!(f16, f32, 0), 2 => floating!(bf16, f32, 0),
            3 | 8 => integral!(u8), 4 => integral!(i64), 5 => integral!(i32),
            6 => integral!(i16), 7 => integral!(i8), 9 => floating!(f64, f64, 9),
            10 => complex!(f32, 0), 11 => complex!(f64, 9), _ => unreachable!(),
        }
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
    }
    finish_dispatch(&c);
}
