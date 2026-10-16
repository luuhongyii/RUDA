use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn indexed_offset(
    indices: &Sequence<Tensor<i64>>, sizes: &Sequence<u32>, strides: &Sequence<u32>,
    position: usize, #[comptime] wrap: bool,
) -> usize {
    let mut offset = 0usize;
    #[unroll]
    for axis in 0..indices.len() {
        let index = indices.index(axis);
        let mut coordinate = index[address(index, position)];
        let size = i64::cast_from(*sizes.index(axis));
        if comptime!(wrap) { coordinate %= size; }
        if coordinate < 0i64 { coordinate += size; }
        offset += usize::cast_from(coordinate) * usize::cast_from(*strides.index(axis));
    }
    offset
}

#[ruda(launch)]
fn index_keys(
    indices: &Sequence<Tensor<i64>>, sizes: &Sequence<u32>, strides: &Sequence<u32>,
    keys: &mut Tensor<i64>, #[comptime] wrap: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < keys.len() { keys[p] = i64::cast_from(indexed_offset(indices, sizes, strides, p, wrap)); }
}

#[ruda(launch)]
fn assign<Q: Numeric + RudaElement>(
    values: &Tensor<f32>, layout: &Tensor<Q>, output: &mut Tensor<Q>,
    indices: &Sequence<Tensor<i64>>, sizes: &Sequence<u32>, strides: &Sequence<u32>,
    keys: &Tensor<i64>, order: &Tensor<u32>, broadcast: usize, tail: usize,
    scale: f64, zero: i64, minimum: i64, maximum: i64,
    #[comptime] sorted: bool, #[comptime] wrap: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < values.len() {
        let position = p / tail % broadcast;
        let mut original = position; let mut offset = 0usize; let mut last = true;
        if comptime!(sorted) {
            original = order[position] as usize;
            offset = usize::cast_from(keys[original]);
            if position + 1usize < broadcast {
                last = keys[original] != keys[order[position + 1usize] as usize];
            }
        } else { offset = indexed_offset(indices, sizes, strides, position, wrap); }
        if last {
            let from = (p / (broadcast * tail) * broadcast + original) * tail + p % tail;
            let value = values[address(values, from)];
            let inverse = 1.0f32 / f32::cast_from(scale);
            let mut q = i64::cast_from(f32::cast_from(zero) + (value * inverse).round());
            if q < minimum { q = minimum; }
            if q > maximum { q = maximum; }
            output[address(layout, p) + offset] = Q::cast_from(q);
        }
    }
}

fn grid(work: usize) -> RudaCount {
    let blocks = work.div_ceil(256); let x = blocks.min(131072);
    RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1)
}

fn span(view: &View) -> usize {
    if view.len == 0 { return 0; }
    view.shape.iter().zip(&view.strides).try_fold(1usize, |n, (&size, &stride)| {
        (size - 1).checked_mul(stride).and_then(|term| n.checked_add(term))
    }).expect("quantized assignment span overflow")
}

fn scratch(c: &ComputeClient<CudaRuntime>, count: usize, dtype: u32) -> View {
    View::packed(c.empty(count.checked_mul(element_bytes(dtype)).expect("quantized assignment workspace overflow")), vec![count], dtype)
}

pub(super) fn dispatch(xs: &[View], ints: &[i64], values: &[f64]) {
    assert!(xs.len() >= 3); let count = xs.len() - 3;
    assert_eq!(ints.len(), 5 + count * 2); assert_eq!(values.len(), 1);
    assert!(matches!(ints[1], 0 | 1) && matches!(ints[2], 0 | 1));
    let (output, layout, input) = (&xs[0], &xs[1], &xs[2]);
    assert_eq!(input.dtype, 0); assert_eq!(output.dtype, layout.dtype); assert_eq!(input.shape, layout.shape);
    let (minimum, maximum): (i64, i64) = match output.dtype {
        5 => (i32::MIN as i64, i32::MAX as i64), 7 => (-128,127), 8 => (0,255),
        _ => panic!("unsupported quantized assignment payload"),
    };
    assert!((minimum..=maximum).contains(&ints[0]));
    let before = usize::try_from(ints[3]).expect("negative leading index dimension count");
    let rank = usize::try_from(ints[4]).expect("negative broadcast rank");
    assert!(before + rank <= layout.shape.len());
    assert!(layout.strides[before..before + rank].iter().all(|&stride| stride == 0));
    let broadcast_shape = &layout.shape[before..before + rank];
    let broadcast = broadcast_shape.iter().product::<usize>();
    let tail = layout.shape[before + rank..].iter().product::<usize>();
    let normalized_shape = if rank == 0 { vec![1] } else { broadcast_shape.to_vec() };
    let indices = &xs[3..];
    let mut dimensions = Vec::with_capacity(count); let mut index_strides = Vec::with_capacity(count);
    for (axis, index) in indices.iter().enumerate() {
        assert_eq!(index.dtype, 4); assert_eq!(index.shape, normalized_shape);
        let size = u32::try_from(ints[5 + axis]).expect("invalid indexed dimension");
        let stride = u32::try_from(ints[5 + count + axis]).expect("invalid indexed stride");
        dimensions.push(size); index_strides.push(stride);
    }
    if input.len == 0 { return; }
    assert!(broadcast > 0 && tail > 0 && output.len > 0);
    let mut maximum_span = span(layout);
    for (&size, &stride) in dimensions.iter().zip(&index_strides) {
        assert!(size > 0);
        maximum_span = maximum_span.checked_add((size as usize - 1).checked_mul(stride as usize)
            .expect("quantized assignment stride overflow")).expect("quantized assignment span overflow");
    }
    assert!(maximum_span <= span(output), "quantized assignment exceeds destination span");
    let deterministic = ints[1] != 0;
    let wrap = deterministic && ints[2] != 0;
    if !wrap {
        for (index, &size) in indices.iter().zip(&dimensions) {
            primitives::check_indices(index, size as usize, true);
        }
    }
    let sorted = deterministic && count > 0 && broadcast > 1;
    let c = client(); let block = RudaDim::new_1d(256);
    let keys = scratch(&c, if sorted { broadcast } else { 1 }, 4);
    let order = if sorted { scratch(&c, broadcast, 5) }
        else { View::packed(keys.handle.clone(), vec![1], 5) };
    let index_args = || indices.iter().map(|index| index.arg()).collect::<SequenceArg<CudaRuntime, Tensor<i64>>>();
    let size_args = || dimensions.iter().copied().collect::<SequenceArg<CudaRuntime, u32>>();
    let stride_args = || index_strides.iter().copied().collect::<SequenceArg<CudaRuntime, u32>>();
    let mut launches = 0u64;
    let _sort_workspace = if sorted {
        unsafe { index_keys::launch::<CudaRuntime>(&c,grid(broadcast),block,index_args(),size_args(),stride_args(),keys.arg(),wrap); }
        let (temporary, sorting_launches) = unique_ops::stable_order(&c,&keys,&order,1,true);
        launches += 1 + sorting_launches;
        Some(temporary)
    } else { None };
    macro_rules! run { ($q:ty) => { unsafe {
        assign::launch::<$q,CudaRuntime>(&c,grid(input.len),block,input.arg(),layout.arg(),output.arg(),
            index_args(),size_args(),stride_args(),keys.arg(),order.arg(),broadcast,tail,
            values[0],ints[0],minimum,maximum,sorted,wrap);
    } }; }
    match output.dtype { 5=>run!(i32),7=>run!(i8),8=>run!(u8),_=>unreachable!() }
    LAUNCHES.fetch_add(launches + 1, Ordering::Relaxed);
}
