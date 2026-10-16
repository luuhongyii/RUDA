use super::*;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn real_value<F: Float>(x: F, y: F, z: F, alpha: F, #[comptime] op: u32) -> F {
    let one = F::new(1.0);
    let mut result = F::new(0.0);
    if comptime!(op == 0) {
        if alpha == one { result = fma(y, z, x); }
        else { result = fma(alpha, y * z, x); }
    } else if comptime!(op == 1) {
        if alpha == one { result = x + y / z; }
        else { result = fma(alpha, y / z, x); }
    } else {
        if z.abs() < F::new(0.5) { result = x + z * (y - x); }
        else { result = y - (y - x) * (one - z); }
    }
    result
}

#[ruda]
fn complex_value<F: Float>(
    x: F, xi: F, y: F, yi: F, z: F, zi: F, alpha: F, alphai: F, #[comptime] op: u32,
) -> (F, F) {
    let zero = F::new(0.0);
    let one = F::new(1.0);
    let mut real = zero;
    let mut imag = zero;
    if comptime!(op == 2) {
        let dr = y - x;
        let di = yi - xi;
        if z * z + zi * zi < F::new(0.25) {
            real = x + (z * dr - zi * di);
            imag = xi + (z * di + zi * dr);
        } else {
            let complement = one - z;
            let complementi = zero - zi;
            real = y - (dr * complement - di * complementi);
            imag = yi - (dr * complementi + di * complement);
        }
    } else {
        let mut product = zero;
        let mut producti = zero;
        if comptime!(op == 0) {
            product = y * z - yi * zi;
            producti = y * zi + yi * z;
        } else {
            let (r, i) = crate::foreach_unary::complex_divide::<F>(y, yi, z, zi);
            product = r; producti = i;
        }
        if alpha == one && alphai == zero {
            real = x + product;
            imag = xi + producti;
        } else {
            real = x + (alpha * product - alphai * producti);
            imag = xi + (alpha * producti + alphai * product);
        }
    }
    (real, imag)
}

#[ruda(launch)]
fn floating<S: Float + RudaElement, F: Float>(
    inputs: &Sequence<Tensor<S>>, first_inputs: &Sequence<Tensor<S>>, second_inputs: &Sequence<Tensor<S>>,
    outputs: &mut Sequence<Tensor<S>>, scalars: &Sequence<u64>,
    #[comptime] op: u32, #[comptime] mode: u32, #[comptime] code: u32,
) {
    let block = RUDA_POS_X as usize;
    let mut first = 0usize;
    #[unroll]
    for tensor in 0..inputs.len() {
        let input = inputs.index(tensor);
        let blocks = input.len() / 65536usize + usize::cast_from(input.len() % 65536usize != 0);
        if block >= first && block < first + blocks {
            let input1 = first_inputs.index(tensor);
            let input2 = second_inputs.index(tensor);
            let output = outputs.index_mut(tensor);
            let scalar = crate::floating::from_bits::<F>(*scalars.index(tensor), code);
            let mut scalar1 = F::new(0.0);
            if comptime!(mode == 1) { scalar1 = F::cast_from(input1[0usize]); }
            let base = (block - first) * 65536usize;
            let remaining = input.len() - base;
            let mut offset = UNIT_POS as usize;
            while offset < remaining && offset < 65536usize {
                let p = base + offset;
                let x = F::cast_from(input[p]);
                let mut y = scalar1;
                if comptime!(mode != 1) { y = F::cast_from(input1[p]); }
                let mut z = scalar;
                if comptime!(mode != 2) { z = F::cast_from(input2[p]); }
                output[p] = S::cast_from(real_value::<F>(x, y, z, scalar, op));
                offset += 512usize;
            }
        }
        first += blocks;
    }
}

#[ruda(launch)]
fn complex<F: Float + RudaElement>(
    inputs: &Sequence<Tensor<F>>, first_inputs: &Sequence<Tensor<F>>, second_inputs: &Sequence<Tensor<F>>,
    outputs: &mut Sequence<Tensor<F>>, scalars: &Sequence<u64>, imag_scalars: &Sequence<u64>,
    #[comptime] op: u32, #[comptime] mode: u32, #[comptime] code: u32,
) {
    let block = RUDA_POS_X as usize;
    let mut first = 0usize;
    #[unroll]
    for tensor in 0..inputs.len() {
        let input = inputs.index(tensor);
        let len = input.len() / 2usize;
        let blocks = len / 65536usize + usize::cast_from(len % 65536usize != 0);
        if block >= first && block < first + blocks {
            let input1 = first_inputs.index(tensor);
            let input2 = second_inputs.index(tensor);
            let output = outputs.index_mut(tensor);
            let scalar = crate::floating::from_bits::<F>(*scalars.index(tensor), code);
            let scalar_imag = crate::floating::from_bits::<F>(*imag_scalars.index(tensor), code);
            let mut scalar1 = F::new(0.0);
            let mut scalar1_imag = F::new(0.0);
            if comptime!(mode == 1) { scalar1 = input1[0usize]; scalar1_imag = input1[1usize]; }
            let base = (block - first) * 65536usize;
            let remaining = len - base;
            let mut offset = UNIT_POS as usize;
            while offset < remaining && offset < 65536usize {
                let p = (base + offset) * 2usize;
                let x = input[p];
                let xi = input[p + 1usize];
                let mut y = scalar1;
                let mut yi = scalar1_imag;
                if comptime!(mode != 1) { y = input1[p]; yi = input1[p + 1usize]; }
                let mut z = scalar;
                let mut zi = scalar_imag;
                if comptime!(mode != 2) { z = input2[p]; zi = input2[p + 1usize]; }
                let (real, imag) = complex_value::<F>(x, xi, y, yi, z, zi, scalar, scalar_imag, op);
                output[p] = real;
                output[p + 1usize] = imag;
                offset += 512usize;
            }
        }
        first += blocks;
    }
}

pub(super) fn dispatch(xs: &[View], ints: &[i64]) {
    assert!(ints.len() >= 3);
    assert!((0..=2).contains(&ints[0]) && (0..=3).contains(&ints[1]) && ints[2] > 0);
    let op = ints[0] as u32;
    let mode = ints[1] as u32;
    assert!((op <= 1 && mode <= 1) || (op == 2 && mode >= 2));
    let n = usize::try_from(ints[2]).expect("foreach ternary tensor count overflow");
    assert_eq!(xs.len(), n.checked_mul(4).expect("foreach ternary tensor count overflow"));
    assert_eq!(ints.len(), n.checked_mul(2).and_then(|v| v.checked_add(3)).expect("foreach ternary scalar count overflow"));
    let code = xs[0].dtype;
    assert!(matches!(code, 0 | 1 | 2 | 9 | 10 | 11));
    for i in 0..n {
        for list in 0..4 {
            let x = &xs[list * n + i];
            let len = if list == 1 && mode == 1 { 1 } else { xs[i].len };
            assert_eq!(x.dtype, code);
            assert_eq!(x.shape, vec![len]);
            assert_eq!(x.strides, vec![1usize]);
        }
    }
    if xs[..n].iter().all(|x| x.len == 0) { return; }
    let storage: Vec<_> = xs.iter().map(|x| {
        let width = if matches!(code, 10 | 11) { 2 } else { 1 };
        let len = x.len.checked_mul(width).expect("foreach ternary component count overflow");
        let dtype = match code { 10 => 0, 11 => 9, _ => code };
        let view = View { handle: x.handle.clone(), shape: vec![len], strides: vec![1], len, dtype };
        crate::scientific::indexable(&view);
        view
    }).collect();
    let indices: Vec<_> = (0..n).filter(|&i| xs[i].len != 0).collect();
    let c = client();
    for batch in indices.chunks(16) {
        let blocks: usize = batch.iter().map(|&i| xs[i].len.div_ceil(65536)).sum();
        let grid = RudaCount::Static(u32::try_from(blocks).expect("foreach ternary grid overflow"), 1, 1);
        let block = RudaDim::new_1d(512);
        let scalars = batch.iter().map(|&i| ints[3 + i * 2] as u64).collect::<SequenceArg<CudaRuntime, u64>>();
        macro_rules! floating { ($s:ty, $f:ty, $code:expr) => { unsafe {
            let inputs = batch.iter().map(|&i| storage[i].arg()).collect::<SequenceArg<CudaRuntime, Tensor<$s>>>();
            let first = batch.iter().map(|&i| storage[n + i].arg()).collect::<SequenceArg<CudaRuntime, Tensor<$s>>>();
            let second = batch.iter().map(|&i| storage[2 * n + i].arg()).collect::<SequenceArg<CudaRuntime, Tensor<$s>>>();
            let outputs = batch.iter().map(|&i| storage[3 * n + i].arg()).collect::<SequenceArg<CudaRuntime, Tensor<$s>>>();
            floating::launch::<$s, $f, CudaRuntime>(&c, grid, block, inputs, first, second, outputs, scalars, op, mode, $code);
        }}; }
        macro_rules! complex { ($f:ty, $code:expr) => { unsafe {
            let inputs = batch.iter().map(|&i| storage[i].arg()).collect::<SequenceArg<CudaRuntime, Tensor<$f>>>();
            let first = batch.iter().map(|&i| storage[n + i].arg()).collect::<SequenceArg<CudaRuntime, Tensor<$f>>>();
            let second = batch.iter().map(|&i| storage[2 * n + i].arg()).collect::<SequenceArg<CudaRuntime, Tensor<$f>>>();
            let outputs = batch.iter().map(|&i| storage[3 * n + i].arg()).collect::<SequenceArg<CudaRuntime, Tensor<$f>>>();
            let imag_scalars = batch.iter().map(|&i| ints[4 + i * 2] as u64).collect::<SequenceArg<CudaRuntime, u64>>();
            complex::launch::<$f, CudaRuntime>(&c, grid, block, inputs, first, second, outputs, scalars, imag_scalars, op, mode, $code);
        }}; }
        match code {
            0 => floating!(f32, f32, 0), 1 => floating!(f16, f32, 0), 2 => floating!(bf16, f32, 0),
            9 => floating!(f64, f64, 9), 10 => complex!(f32, 0), 11 => complex!(f64, 9), _ => unreachable!(),
        }
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
    }
    finish_dispatch(&c);
}
