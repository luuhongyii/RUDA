use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;
use ruda_kernel::library::FastDivmod;

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_arithmetic_api_version() -> u32 { 3 }

#[ruda(launch)]
fn floating_direct<A: Float, B: Float, S: Float>(
    left: &Tensor<A>, right: &Tensor<B>, output: &mut Tensor<S>, alpha: u64, immediate: u64,
    shape: &Sequence<FastDivmod<usize>>,
    #[comptime] op: u32, #[comptime] scalar_left: bool, #[comptime] scalar_right: bool,
    #[comptime] left_packed: bool, #[comptime] right_packed: bool, #[comptime] output_packed: bool,
    #[comptime] fast_index: bool,
    #[define(A)] _left: StorageType, #[define(B)] _right: StorageType, #[define(S)] _output: StorageType,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < output.len() {
        let mut i = p; let mut j = p; let mut destination = p;
        if comptime!(fast_index) {
            if comptime!(!scalar_left && !left_packed) { i=0; }
            if comptime!(op!=2 && !scalar_right && !right_packed) { j=0; }
            if comptime!(!output_packed) { destination=0; }
            let mut remaining=p;
            #[unroll]
            for index in 0usize..shape.len().comptime() {
                let dim=comptime!(shape.len()-index-1);
                let (quotient,coordinate)=shape[dim].div_mod(remaining);
                remaining=quotient;
                if comptime!(!scalar_left && !left_packed) { i+=coordinate*left.stride(dim); }
                if comptime!(op!=2 && !scalar_right && !right_packed) { j+=coordinate*right.stride(dim); }
                if comptime!(!output_packed) { destination+=coordinate*output.stride(dim); }
            }
        } else {
            if comptime!(!scalar_left && !left_packed) { i=address(left,p); }
            if comptime!(op!=2 && !scalar_right && !right_packed) { j=address(right,p); }
            if comptime!(!output_packed) { destination=address(output,p); }
        }
        let mut a = f32::reinterpret(u32::cast_from(immediate));
        if comptime!(!scalar_left) {
            a = f32::cast_from(left[i]);
        }
        let mut result = -a;
        if comptime!(op != 2) {
            let mut b = f32::reinterpret(u32::cast_from(immediate));
            if comptime!(!scalar_right) {
                b = f32::cast_from(right[j]);
            }
            if comptime!(op == 0) { result = fma(f32::reinterpret(u32::cast_from(alpha)), b, a); }
            else { result = a * b; }
        }
        output[destination] = S::cast_from(result);
    }
}

#[ruda(launch)]
fn floating_vector<A: Float, B: Float, S: Float>(
    left: &Array<Vector<A, Const<4>>>, right: &Array<Vector<B, Const<4>>>,
    output: &mut Array<Vector<S, Const<4>>>, alpha: u64, immediate: u64,
    #[comptime] op: u32, #[comptime] scalar_left: bool, #[comptime] scalar_right: bool,
    #[define(A)] _left: StorageType, #[define(B)] _right: StorageType, #[define(S)] _output: StorageType,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < output.len() {
        let mut a = Vector::<f32, Const<4>>::new(f32::reinterpret(u32::cast_from(immediate)));
        let mut b = a;
        if comptime!(!scalar_left) { a = Vector::cast_from(left[p]); }
        if comptime!(op != 2 && !scalar_right) { b = Vector::cast_from(right[p]); }
        let mut result = Vector::<S, Const<4>>::new(S::cast_from(0.0f32));
        #[unroll]
        for lane in 0usize..4 {
            let mut value = -a[lane];
            if comptime!(op == 0) { value = fma(f32::reinterpret(u32::cast_from(alpha)), b[lane], a[lane]); }
            else if comptime!(op == 1) { value = a[lane] * b[lane]; }
            result[lane] = S::cast_from(value);
        }
        output[p] = result;
    }
}

fn direct(xs: &[View], ints: &[i64]) {
    assert!((0..=2).contains(&ints[0]) && matches!(ints[1], 0 | 2) && matches!(ints[2], 0 | 2));
    assert!(xs.iter().all(|x| x.dtype <= 2));
    let op = ints[0] as u32;
    let (scalar_left, scalar_right) = (ints[1] == 2, ints[2] == 2);
    assert!(ints[1] + ints[2] <= 2 && (op != 2 || !scalar_left && !scalar_right));
    assert_eq!(ints.len(), if scalar_left || scalar_right { 5 } else { 4 });
    for (input, scalar) in [(&xs[0], scalar_left), (&xs[1], scalar_right)] {
        if !scalar { assert_eq!(input.shape, xs[2].shape); }
    }
    if xs[2].len == 0 { return; }
    let left_packed=scalar_left || xs[0].contiguous();
    let right_packed=scalar_right || xs[1].contiguous();
    let output_packed=xs[2].contiguous();
    let contiguous=left_packed && right_packed && output_packed;
    let dtype = |code| -> StorageType {
        (match code { 0 => ruda_core::tensor::DType::F32, 1 => ruda_core::tensor::DType::F16,
            2 => ruda_core::tensor::DType::BF16, _ => unreachable!() }).into()
    };
    let vector = contiguous && xs[2].len >= 1024 && xs[2].len % 4 == 0;
    let c = client(); let blocks = (xs[2].len / if vector { 4 } else { 1 }).div_ceil(256); let x = blocks.min(131072);
    let grid = RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1);
    unsafe {
        if vector {
            floating_vector::launch::<CudaRuntime>(&c, grid, RudaDim::new_1d(256),
                xs[0].array_arg(), xs[1].array_arg(), xs[2].array_arg(), ints[3] as u64, ints.get(4).copied().unwrap_or(0) as u64,
                op, scalar_left, scalar_right, dtype(xs[0].dtype), dtype(xs[1].dtype), dtype(xs[2].dtype));
        } else {
            let fast_index=!contiguous && xs[2].len<=i32::MAX as usize;
            let mut shape=SequenceArg::<CudaRuntime,FastDivmod<usize>>::new();
            if fast_index { for &size in &xs[2].shape { shape.push(size); } }
            floating_direct::launch::<CudaRuntime>(&c, grid, RudaDim::new_1d(256),
                xs[0].arg(), xs[1].arg(), xs[2].arg(), ints[3] as u64, ints.get(4).copied().unwrap_or(0) as u64,
                shape, op, scalar_left, scalar_right, left_packed, right_packed, output_packed, fast_index,
                dtype(xs[0].dtype), dtype(xs[1].dtype), dtype(xs[2].dtype));
        }
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed); finish_dispatch(&c);
}

#[ruda(launch_unchecked)]
fn contiguous<S: Float + RudaElement, F: Float>(
    left: &Array<Vector<S, Const<4>>>, right: &Array<Vector<S, Const<4>>>,
    output: &mut Array<Vector<S, Const<4>>>, alpha: u64,
    #[comptime] op: u32, #[comptime] code: u32,
) {
    let pos = ABSOLUTE_POS as usize;
    if pos < output.len() {
        let a = left[pos];
        let mut result = Vector::<S, Const<4>>::empty();
        if comptime!(op == 2) {
            #[unroll]
            for i in 0usize..4 { result[i] = S::cast_from(-F::cast_from(a[i])); }
        } else {
            let b = right[pos];
            #[unroll]
            for i in 0usize..4 {
                if comptime!(op == 0) {
                    result[i] = S::cast_from(fma(crate::floating::from_bits::<F>(alpha, code), F::cast_from(b[i]), F::cast_from(a[i])));
                } else { result[i] = S::cast_from(F::cast_from(a[i]) * F::cast_from(b[i])); }
            }
        }
        output[pos] = result;
    }
}

#[ruda(launch)]
fn floating<A: Float + RudaElement, B: Float + RudaElement, S: Float + RudaElement, F: Float>(
    left: &Tensor<A>, right: &Tensor<B>, output: &mut Tensor<S>, alpha: u64,
    #[comptime] op: u32, #[comptime] scalar_left: bool, #[comptime] scalar_right: bool, #[comptime] code: u32,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < output.len() {
        let mut i = 0usize; let mut j = 0usize;
        if comptime!(!scalar_left) { i = address(left, p); }
        let a = F::cast_from(left[i]); let mut result = -a;
        if comptime!(op != 2) {
            if comptime!(!scalar_right) { j = address(right, p); }
            let b = F::cast_from(right[j]);
            if comptime!(op == 0) { result = fma(crate::floating::from_bits::<F>(alpha, code), b, a); }
            else { result = a * b; }
        }
        output[address(output, p)] = S::cast_from(result);
    }
}

#[ruda(launch)]
fn integral<S: Numeric + RudaElement>(
    left: &Tensor<S>, right: &Tensor<S>, output: &mut Tensor<S>, alpha: u64,
    #[comptime] op: u32, #[comptime] scalar_left: bool, #[comptime] scalar_right: bool, #[comptime] boolean: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < output.len() {
        let mut i = 0usize; let mut j = 0usize;
        if comptime!(!scalar_left) { i = address(left, p); }
        let a = u64::cast_from(left[i]); let mut result = 0u64 - a;
        if comptime!(op != 2) {
            if comptime!(!scalar_right) { j = address(right, p); }
            let b = u64::cast_from(right[j]);
            if comptime!(boolean) {
                if comptime!(op == 0) { result = u64::cast_from(a != 0u64 || alpha != 0u64 && b != 0u64); }
                else { result = u64::cast_from(a != 0u64 && b != 0u64); }
            } else if comptime!(op == 0) { result = a + alpha * b; }
            else { result = a * b; }
        }
        output[address(output, p)] = S::cast_from(result);
    }
}

pub(super) fn dispatch(xs: &[View], ints: &[i64]) {
    assert_eq!(xs.len(), 3);
    if (ints.len() == 5 || ints.len() == 4 && ints[1] == 0 && ints[2] == 0)
        && xs.iter().all(|x| x.dtype <= 2) {
        let packed = ints.len() == 4 && xs[2].len % 4 == 0 && xs[2].len <= u32::MAX as usize
            && xs.iter().all(|v| v.dtype == xs[2].dtype && v.is_contiguous()
                && v.handle.offset_start.unwrap_or(0) % (4 * element_bytes(v.dtype)) as u64 == 0);
        if !packed { direct(xs, ints); return; }
    }
    assert_eq!(ints.len(), 4);
    assert!((0..=2).contains(&ints[0]) && matches!(ints[1], 0 | 1) && matches!(ints[2], 0 | 1));
    let op = ints[0] as u32; let scalar_left = ints[1] != 0; let scalar_right = ints[2] != 0;
    assert!(ints[1] + ints[2] <= 1); assert!(op != 2 || !scalar_left && !scalar_right);
    let code = xs[2].dtype; assert!(code <= 9 && (op != 2 || code != 3));
    let math_code = if matches!(code, 1 | 2) { 0 } else { code };
    for (input, scalar) in [(&xs[0], scalar_left), (&xs[1], scalar_right)] {
        assert_eq!(input.dtype, if scalar { math_code } else { code });
        if scalar { assert_eq!(input.len, 1); } else { assert_eq!(input.shape, xs[2].shape); }
    }
    if xs[2].len == 0 { return; }
    let c = client(); let blocks = xs[2].len.div_ceil(256); let x = blocks.min(131072);
    let grid = RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1); let block = RudaDim::new_1d(256);
    let alpha = ints[3] as u64;
    let packed = code <= 2 && !scalar_left && !scalar_right && xs[2].len % 4 == 0
        && xs[2].len <= u32::MAX as usize && xs.iter().all(|v| v.is_contiguous()
            && v.handle.offset_start.unwrap_or(0) % (4 * element_bytes(v.dtype)) as u64 == 0);
    macro_rules! real { ($s:ty, $f:ty) => { unsafe {
        if packed {
            contiguous::launch_unchecked::<$s, $f, CudaRuntime>(&c,
                RudaCount::Static(xs[2].len.div_ceil(512) as u32, 1, 1), RudaDim::new_1d(128),
                primitives::tensor(&xs[0]).into_array_arg(), primitives::tensor(&xs[1]).into_array_arg(),
                primitives::tensor(&xs[2]).into_array_arg(), alpha, op, math_code);
        } else if scalar_left {
            floating::launch::<$f, $s, $s, $f, CudaRuntime>(&c, grid, block, xs[0].arg(), xs[1].arg(), xs[2].arg(), alpha, op, true, false, math_code);
        } else if scalar_right {
            floating::launch::<$s, $f, $s, $f, CudaRuntime>(&c, grid, block, xs[0].arg(), xs[1].arg(), xs[2].arg(), alpha, op, false, true, math_code);
        } else {
            floating::launch::<$s, $s, $s, $f, CudaRuntime>(&c, grid, block, xs[0].arg(), xs[1].arg(), xs[2].arg(), alpha, op, false, false, math_code);
        }
    }}; }
    macro_rules! integer { ($s:ty) => { unsafe {
        integral::launch::<$s, CudaRuntime>(&c, grid, block, xs[0].arg(), xs[1].arg(), xs[2].arg(), alpha, op, scalar_left, scalar_right, code == 3);
    }}; }
    match code {
        0 => real!(f32, f32), 1 => real!(f16, f32), 2 => real!(bf16, f32), 9 => real!(f64, f64),
        3 | 8 => integer!(u8), 4 => integer!(i64), 5 => integer!(i32), 6 => integer!(i16), 7 => integer!(i8), _ => unreachable!(),
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed); finish_dispatch(&c);
}
