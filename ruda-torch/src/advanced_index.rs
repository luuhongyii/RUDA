use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn normalize(raw: i64, size: u32, #[comptime] clamp: bool) -> i64 {
    let limit = i64::cast_from(size); let mut index = raw;
    if comptime!(clamp) {
        if index < -limit { index = -limit; }
        if index >= limit { index = limit - 1i64; }
    }
    if index < 0i64 { index += limit; }
    if index < 0i64 || index >= limit { index = -1i64; }
    if comptime!(clamp) { if size == 0u32 { index = 0i64; } }
    index
}

#[ruda(launch)]
fn offsets(
    indices32: &Sequence<Tensor<i32>>, indices64: &Sequence<Tensor<i64>>, output: &mut Tensor<i64>,
    sizes32: &Sequence<u32>, strides32: &Sequence<u32>, sizes64: &Sequence<u32>, strides64: &Sequence<u32>,
    free_sizes: &Sequence<u32>, free_strides: &Sequence<u32>, free_divisors: &Sequence<u32>,
    broadcast_length: usize, tail: usize, #[comptime] clamp: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < output.len() {
        let mut offset = 0usize; let mut valid = true;
        #[unroll]
        for dim in 0..free_sizes.len() {
            let coordinate = p / usize::cast_from(*free_divisors.index(dim)) % usize::cast_from(*free_sizes.index(dim));
            offset += coordinate * usize::cast_from(*free_strides.index(dim));
        }
        let at = p / tail % broadcast_length;
        #[unroll]
        for dim in 0..indices32.len() {
            let index = indices32.index(dim);
            let coordinate = normalize(i64::cast_from(index[address(index, at)]), *sizes32.index(dim), clamp);
            if coordinate < 0i64 { valid = false; }
            else { offset += usize::cast_from(coordinate) * usize::cast_from(*strides32.index(dim)); }
        }
        #[unroll]
        for dim in 0..indices64.len() {
            let index = indices64.index(dim);
            let coordinate = normalize(index[address(index, at)], *sizes64.index(dim), clamp);
            if coordinate < 0i64 { valid = false; }
            else { offset += usize::cast_from(coordinate) * usize::cast_from(*strides64.index(dim)); }
        }
        let mut result = -1i64;
        if valid { result = i64::cast_from(offset); }
        output[address(output, p)] = result;
    }
}

fn product(shape: &[usize]) -> usize {
    shape.iter().try_fold(1usize, |n, &size| n.checked_mul(size)).expect("advanced index shape overflow")
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut result = vec![1; shape.len()]; let mut stride = 1usize;
    for dim in (0..shape.len()).rev() {
        result[dim] = stride; stride = stride.checked_mul(shape[dim]).expect("advanced index stride overflow");
    }
    result
}

fn parameters(values: impl IntoIterator<Item = usize>) -> SequenceArg<CudaRuntime, u32> {
    values.into_iter().map(|value| u32::try_from(value).expect("advanced index exceeds U32 indexing")).collect()
}

pub(super) fn dispatch(xs: &[View], ints: &[i64]) {
    assert!(!xs.is_empty() && ints.len() >= 3);
    assert!(ints[0] >= 0 && ints[1] >= 0 && matches!(ints[2], 0 | 1));
    let rank = ints[0] as usize; let broadcast_rank = ints[1] as usize; let clamp = ints[2] != 0;
    let output = &xs[0]; let inputs = &xs[1..];
    assert_eq!(output.dtype, 4); assert_eq!(ints.len(), 3 + rank + inputs.len());
    let shape = ints[3..3 + rank].iter().map(|&size| {
        assert!(size >= 0 && size <= u32::MAX as i64); size as usize
    }).collect::<Vec<_>>();
    let axes = ints[3 + rank..].iter().map(|&axis| {
        assert!(axis >= 0 && (axis as usize) < rank); axis as usize
    }).collect::<Vec<_>>();
    assert!(axes.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(inputs.iter().all(|index| matches!(index.dtype, 4 | 5)));
    let broadcast_shape = if broadcast_rank == 0 { vec![] } else {
        assert!(!inputs.is_empty()); assert_eq!(inputs[0].shape.len(), broadcast_rank); inputs[0].shape.clone()
    };
    let normalized_broadcast = if broadcast_shape.is_empty() { vec![1] } else { broadcast_shape.clone() };
    assert!(inputs.iter().all(|index| index.shape == normalized_broadcast));
    let contiguous = !axes.is_empty() && axes.windows(2).all(|pair| pair[1] == pair[0] + 1);
    let remaining = (0..rank).filter(|dim| !axes.contains(dim)).collect::<Vec<_>>();
    let before = remaining.iter().copied().filter(|&dim| contiguous && dim < axes[0]).collect::<Vec<_>>();
    let after = remaining.iter().copied().filter(|dim| !before.contains(dim)).collect::<Vec<_>>();
    let mut expected = before.iter().map(|&dim| shape[dim]).collect::<Vec<_>>();
    expected.extend(&broadcast_shape); expected.extend(after.iter().map(|&dim| shape[dim]));
    let normalized_output = if expected.is_empty() { vec![1] } else { expected.clone() };
    assert_eq!(output.shape, normalized_output);
    if output.len == 0 { return; }
    assert!(product(&shape) <= u32::MAX as usize, "advanced index input exceeds U32 indexing");
    let input_strides = strides(&shape); let output_strides = strides(&expected);
    let free = before.iter().chain(&after).copied().collect::<Vec<_>>();
    let free_sizes = parameters(free.iter().map(|&dim| shape[dim]));
    let free_strides = parameters(free.iter().map(|&dim| input_strides[dim]));
    let free_divisors = parameters((0..free.len()).map(|position| {
        output_strides[if position < before.len() { position } else { position + broadcast_rank }]
    }));
    let group32 = (0..inputs.len()).filter(|&i| inputs[i].dtype == 5).collect::<Vec<_>>();
    let group64 = (0..inputs.len()).filter(|&i| inputs[i].dtype == 4).collect::<Vec<_>>();
    let indices32 = group32.iter().map(|&i| inputs[i].arg()).collect::<SequenceArg<CudaRuntime, Tensor<i32>>>();
    let indices64 = group64.iter().map(|&i| inputs[i].arg()).collect::<SequenceArg<CudaRuntime, Tensor<i64>>>();
    let sizes32 = parameters(group32.iter().map(|&i| shape[axes[i]]));
    let strides32 = parameters(group32.iter().map(|&i| input_strides[axes[i]]));
    let sizes64 = parameters(group64.iter().map(|&i| shape[axes[i]]));
    let strides64 = parameters(group64.iter().map(|&i| input_strides[axes[i]]));
    let broadcast_length = product(&broadcast_shape);
    let tail = product(&after.iter().map(|&dim| shape[dim]).collect::<Vec<_>>());
    let blocks = output.len.div_ceil(256); let x = blocks.min(131072);
    let c = client();
    unsafe { offsets::launch::<CudaRuntime>(&c, RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1), RudaDim::new_1d(256),
        indices32, indices64, output.arg(), sizes32, strides32, sizes64, strides64, free_sizes, free_strides, free_divisors,
        broadcast_length, tail, clamp); }
    LAUNCHES.fetch_add(1, Ordering::Relaxed); finish_dispatch(&c);
}
