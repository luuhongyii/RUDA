use super::*;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn maximum<N: Numeric>(a: N, b: N) -> N {
    let mut result = a;
    if b != b || a < b { result = b; }
    result
}

#[ruda]
fn combine<N: Numeric>(a: N, b: N, #[comptime] is_max: bool) -> N {
    let mut result = N::cast_from(0u32);
    if comptime!(is_max) { result = maximum::<N>(a, b); }
    else { result = a + b; }
    result
}

#[ruda]
fn block_reduce<N: Numeric>(value: N, #[comptime] is_max: bool) -> N {
    let tid = UNIT_POS as usize;
    let lane = tid % 32usize;
    let warp = tid / 32usize;
    let mut result = value;
    let mut offset = 16u32;
    while offset > 0u32 {
        result = combine::<N>(result, plane_shuffle_down(result, offset), is_max);
        offset /= 2u32;
    }
    let mut shared = SharedMemory::<N>::new(16usize);
    sync_ruda();
    if lane == 0 { shared[warp] = result; }
    sync_ruda();
    result = N::cast_from(0u32);
    if comptime!(is_max) { result = N::min_value(); }
    if tid < 16usize { result = shared[lane]; }
    if warp == 0 {
        offset = 16u32;
        while offset > 0u32 {
            result = combine::<N>(result, plane_shuffle_down(result, offset), is_max);
            offset /= 2u32;
        }
    }
    result
}

#[ruda]
fn accumulate<N: Numeric>(sum: N, value: N, #[comptime] kind: u32) -> N {
    let mut result = N::cast_from(0u32);
    if comptime!(kind == 0) { result = maximum::<N>(sum, value); }
    else if comptime!(kind == 1) { result = sum + N::cast_from(value != N::cast_from(0u32)); }
    else if comptime!(kind == 2) { result = sum + value.abs(); }
    else if comptime!(kind == 3) { result = sum + value * value; }
    else { result = maximum::<N>(sum, value.abs()); }
    result
}

#[ruda(launch)]
fn chunks<S: Numeric + RudaElement, A: Numeric + RudaElement>(
    inputs: &Sequence<Tensor<S>>, partials: &mut Tensor<A>, offsets: &Sequence<u32>, aligned: &Sequence<u32>,
    #[comptime] kind: u32, #[comptime] boolean: bool,
) {
    let block = RUDA_POS_X as usize;
    let tid = UNIT_POS as usize;
    let mut first = 0usize;
    #[unroll]
    for tensor in 0..inputs.len() {
        let input = inputs.index(tensor);
        let blocks = input.len() / 65536usize + usize::cast_from(input.len() % 65536usize != 0);
        if block >= first && block < first + blocks {
            let chunk = block - first;
            let base = chunk * 65536usize;
            let remaining = input.len() - base;
            let packed = *aligned.index(tensor) != 0u32 && remaining % 4usize == 0;
            let mut identity = A::cast_from(0u32);
            if comptime!(kind == 0) { identity = A::min_value(); }
            let mut values = Array::<A>::new(4usize);
            #[unroll]
            for ilp in 0..4usize { values[ilp] = identity; }
            let mut start = 0usize;
            while start < remaining && start < 65536usize {
                #[unroll]
                for ilp in 0..4usize {
                    let mut p = start + tid + ilp * 512usize;
                    if packed { p = start + tid * 4usize + ilp; }
                    if p < remaining && p < 65536usize {
                        let mut value = A::cast_from(input[base + p]);
                        if comptime!(boolean) { value = A::cast_from(value != A::cast_from(0u32)); }
                        values[ilp] = accumulate::<A>(values[ilp], value, kind);
                    }
                }
                start += 2048usize;
            }
            let mut value = identity;
            #[unroll]
            for ilp in 0..4usize { value = combine::<A>(value, values[ilp], kind == 0 || kind == 4); }
            let result = block_reduce::<A>(value, kind == 0 || kind == 4);
            if tid == 0 { partials[usize::cast_from(*offsets.index(tensor)) + chunk] = result; }
        }
        first += blocks;
    }
}

#[ruda(launch)]
fn cleanup<A: Numeric + RudaElement, O: Numeric + RudaElement>(
    partials: &Tensor<A>, outputs: &mut Sequence<Tensor<O>>, offsets: &Sequence<u32>, counts: &Sequence<u32>,
    max_chunks: usize, #[comptime] kind: u32, #[comptime] root: bool, #[comptime] code: u32,
) {
    let block = RUDA_POS_X as usize;
    let tid = UNIT_POS as usize;
    #[unroll]
    for tensor in 0..outputs.len() {
        if block == tensor {
            let output = outputs.index_mut(tensor);
            let first = usize::cast_from(*offsets.index(tensor));
            let count = usize::cast_from(*counts.index(tensor));
            let mut identity = A::cast_from(0u32);
            if comptime!(kind == 0) { identity = A::min_value(); }
            let mut value = identity;
            let mut chunk = tid;
            while chunk < max_chunks {
                let mut part = identity;
                if chunk < count { part = partials[first + chunk]; }
                value = combine::<A>(value, part, kind == 0 || kind == 4);
                chunk += 512usize;
            }
            let mut result = block_reduce::<A>(value, kind == 0 || kind == 4);
            if tid == 0 {
                if comptime!(root) {
                    if comptime!(code == 9) { result = A::cast_from(f64::cast_from(result).sqrt()); }
                    else { result = A::cast_from(f32::cast_from(result).sqrt()); }
                }
                output[0usize] = O::cast_from(result);
            }
        }
    }
}

pub(super) fn dispatch(xs: &[View], ints: &[i64]) {
    assert!(ints.len() >= 3);
    assert!((0..=4).contains(&ints[0]) && (0..=1).contains(&ints[1]) && ints[2] > 0);
    let kind = ints[0] as u32;
    let root = ints[1] != 0;
    assert!(!root || kind == 3);
    let n = usize::try_from(ints[2]).expect("foreach reduce tensor count overflow");
    assert_eq!(xs.len(), n.checked_mul(2).expect("foreach reduce tensor count overflow"));
    assert_eq!(ints.len(), n.checked_add(3).expect("foreach reduce alignment count overflow"));
    let code = xs[0].dtype;
    let output_code = xs[n].dtype;
    if kind == 0 { assert!(code <= 9); assert_eq!(code, output_code); }
    else {
        assert!(matches!(code, 0 | 1 | 2 | 9) && matches!(output_code, 0 | 1 | 2 | 9));
        assert!(code == output_code || output_code == 9 || (matches!(code, 1 | 2) && output_code == 0));
    }
    let partial_code = if kind == 0 { code } else if output_code == 9 { 9 } else { 0 };
    let mut offsets = Vec::with_capacity(n);
    let mut counts = Vec::with_capacity(n);
    let mut total = 0usize;
    let mut max_chunks = 0usize;
    for i in 0..n {
        assert!(matches!(ints[3 + i], 0 | 1));
        assert_eq!(xs[i].dtype, code);
        assert_eq!(xs[i].shape, vec![xs[i].len]);
        assert_eq!(xs[i].strides, vec![1usize]);
        if kind == 0 || kind == 4 { assert!(xs[i].len != 0); }
        assert_eq!(xs[n + i].dtype, output_code);
        assert_eq!(xs[n + i].shape, vec![1usize]);
        assert_eq!(xs[n + i].strides, vec![1usize]);
        crate::scientific::indexable(&xs[i]);
        let count = xs[i].len.div_ceil(65536);
        offsets.push(u32::try_from(total).expect("foreach reduce workspace offset overflow"));
        counts.push(u32::try_from(count).expect("foreach reduce chunk count overflow"));
        total = total.checked_add(count).expect("foreach reduce workspace overflow");
        max_chunks = max_chunks.max(count);
    }
    assert!(total <= u32::MAX as usize, "foreach reduce workspace exceeds U32 indexing");
    let c = client();
    let bytes = total.max(1).checked_mul(element_bytes(partial_code)).expect("foreach reduce byte count overflow");
    let partials = View::packed(c.empty(bytes.max(4)), vec![total.max(1)], partial_code);
    for first in (0..n).step_by(16) {
        let end = (first + 16).min(n);
        let blocks: usize = counts[first..end].iter().map(|&v| v as usize).sum();
        let grid = RudaCount::Static(u32::try_from(blocks).expect("foreach reduce grid overflow"), 1, 1);
        let cleanup_grid = RudaCount::Static((end - first) as u32, 1, 1);
        let block = RudaDim::new_1d(512);
        macro_rules! run { ($s:ty, $a:ty, $o:ty) => { unsafe {
            if blocks != 0 {
                let inputs = (first..end).map(|i| xs[i].arg()).collect::<SequenceArg<CudaRuntime, Tensor<$s>>>();
                let starts = offsets[first..end].iter().copied().collect::<SequenceArg<CudaRuntime, u32>>();
                let aligned = (first..end).map(|i| ints[3 + i] as u32).collect::<SequenceArg<CudaRuntime, u32>>();
                chunks::launch::<$s, $a, CudaRuntime>(&c, grid, block, inputs, partials.arg(), starts, aligned, kind, code == 3);
                LAUNCHES.fetch_add(1, Ordering::Relaxed);
            }
            let outputs = (first..end).map(|i| xs[n + i].arg()).collect::<SequenceArg<CudaRuntime, Tensor<$o>>>();
            let starts = offsets[first..end].iter().copied().collect::<SequenceArg<CudaRuntime, u32>>();
            let sizes = counts[first..end].iter().copied().collect::<SequenceArg<CudaRuntime, u32>>();
            cleanup::launch::<$a, $o, CudaRuntime>(&c, cleanup_grid, block, partials.arg(), outputs, starts, sizes,
                max_chunks, kind, root, partial_code);
            LAUNCHES.fetch_add(1, Ordering::Relaxed);
        }}; }
        macro_rules! norm { ($s:ty) => { match output_code {
            0 => run!($s, f32, f32), 1 => run!($s, f32, f16), 2 => run!($s, f32, bf16),
            9 => run!($s, f64, f64), _ => unreachable!(),
        }}; }
        if kind == 0 {
            match code {
                0 => run!(f32, f32, f32), 1 => run!(f16, f16, f16), 2 => run!(bf16, bf16, bf16),
                3 | 8 => run!(u8, u8, u8), 4 => run!(i64, i64, i64), 5 => run!(i32, i32, i32),
                6 => run!(i16, i16, i16), 7 => run!(i8, i8, i8), 9 => run!(f64, f64, f64), _ => unreachable!(),
            }
        } else {
            match code { 0 => norm!(f32), 1 => norm!(f16), 2 => norm!(bf16), 9 => norm!(f64), _ => unreachable!() }
        }
    }
    finish_dispatch(&c);
}
