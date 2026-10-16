use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::{Tensor, fma};

#[ruda(launch)]
fn clear_status(status: &mut Tensor<u32>) {
    if UNIT_POS == 0u32 { status[0] = 0u32; }
}

#[ruda(launch)]
fn check_indices<I: Int + RudaElement>(indices: &Tensor<I>, status: &mut Tensor<Atomic<u32>>, rows: usize) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < indices.len() {
        let row = i64::cast_from(indices[address(indices, p)]);
        if row < 0i64 || row >= i64::cast_from(rows) { status[0].store(1u32); }
    }
}

#[ruda(launch)]
fn gather<S: Numeric + RudaElement, I: Int + RudaElement>(
    weight: &Tensor<S>, weight_imaginary: &Tensor<S>, indices: &Tensor<I>,
    output: &mut Tensor<S>, output_imaginary: &mut Tensor<S>, #[comptime] complex: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < output.len() {
        let width = weight.shape(1usize);
        let row = usize::cast_from(indices[address(indices, p / width)]);
        let from = row * width + p % width;
        output[address(output, p)] = weight[address(weight, from)];
        if comptime!(complex) { output_imaginary[address(output_imaginary, p)] = weight_imaginary[address(weight_imaginary, from)]; }
    }
}

#[ruda(launch)]
fn zero<S: Numeric + RudaElement>(output: &mut Tensor<S>) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < output.len() { output[address(output, p)] = S::cast_from(0u32); }
}

#[ruda(launch)]
fn small_backward<S: Float + RudaElement, F: Float + RudaElement, I: Int + RudaElement>(
    grad: &Tensor<S>, indices: &Tensor<I>, output: &mut Tensor<S>, padding: i64,
) {
    let block = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    let tid = UNIT_POS as usize; let lane = tid % 32usize; let warp = tid / 32usize;
    let feature = block * 32usize + lane; let width = output.shape(1usize);
    let mut rows = SharedMemory::<i64>::new(32usize);
    let mut values = SharedMemory::<F>::new(1024usize);
    let mut first = 0usize;
    while first < indices.len() {
        let sample = first + warp;
        let mut row = -1i64; let mut value = F::new(0.0);
        if sample < indices.len() {
            row = i64::cast_from(indices[address(indices, sample)]);
            if feature < width && row != padding { value = F::cast_from(grad[address(grad, sample * width + feature)]); }
        }
        if lane == 0usize { rows[warp] = row; }
        values[tid] = value;
        sync_ruda();
        let mut leader = sample < indices.len() && row != padding;
        let mut peer = 0usize;
        while peer < warp {
            if rows[peer] == row { leader = false; }
            peer += 1usize;
        }
        if leader && feature < width {
            let mut sum = values[tid];
            peer = warp + 1usize;
            while peer < 32usize {
                if rows[peer] == row { sum += values[peer * 32usize + lane]; }
                peer += 1usize;
            }
            let at = address(output, usize::cast_from(row) * width + feature);
            let rounded = S::cast_from(sum);
            output[at] = S::cast_from(F::cast_from(output[at]) + F::cast_from(rounded));
        }
        sync_ruda();
        first += 32usize;
    }
}

#[ruda]
fn bound<I: Int>(indices: &Tensor<I>, order: &Tensor<u32>, row: i64, #[comptime] upper: bool) -> usize {
    let mut low = 0usize; let mut high = order.len();
    while low < high {
        let middle = low + (high - low) / 2usize;
        let key = i64::cast_from(indices[address(indices, order[middle] as usize)]);
        let mut advance = key < row;
        if comptime!(upper) { advance = key <= row; }
        if advance { low = middle + 1usize; } else { high = middle; }
    }
    low
}

#[ruda(launch)]
fn chunk_flags<I: Int + RudaElement>(indices: &Tensor<I>, order: &Tensor<u32>, prefix: &mut Tensor<u32>) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < order.len() {
        let row = i64::cast_from(indices[address(indices, order[p] as usize)]);
        let first = bound(indices, order, row, false);
        prefix[p] = u32::cast_from((p - first) % 10usize == 0usize);
    }
}

#[ruda(launch)]
fn chunk_count(prefix: &Tensor<u32>, total: &mut Tensor<u32>) {
    if UNIT_POS == 0u32 { total[0] = prefix[prefix.len() - 1usize]; }
}

#[ruda(launch)]
fn chunk_metadata<I: Int + RudaElement>(
    indices: &Tensor<I>, order: &Tensor<u32>, prefix: &Tensor<u32>, starts: &mut Tensor<u32>, counts: &mut Tensor<u32>,
    #[comptime] frequency: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < prefix.len() {
        let mut first = p == 0usize;
        if p > 0usize { first = prefix[p] != prefix[p - 1usize]; }
        if first {
            let chunk = (prefix[p] - 1u32) as usize;
            starts[chunk] = u32::cast_from(p);
            if comptime!(frequency) {
                let row = i64::cast_from(indices[address(indices, order[p] as usize)]);
                counts[chunk] = u32::cast_from(bound(indices, order, row, true) - bound(indices, order, row, false));
            }
        }
    }
}

#[ruda(launch)]
fn partials<S: Float + RudaElement, F: Float + RudaElement>(
    grad: &Tensor<S>, order: &Tensor<u32>, starts: &Tensor<u32>, counts: &Tensor<u32>, output: &mut Tensor<F>,
    width: usize, #[comptime] frequency: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < output.len() {
        let chunk = p / width; let feature = p % width;
        let first = starts[chunk] as usize;
        let mut end = order.len();
        if chunk + 1usize < starts.len() { end = starts[chunk + 1usize] as usize; }
        let mut scale = F::new(1.0);
        if comptime!(frequency) { scale = F::new(1.0) / F::cast_from(counts[chunk]); }
        let mut value = F::new(0.0); let mut cursor = first;
        while cursor < end {
            let sample = order[cursor] as usize;
            value = fma(F::cast_from(grad[address(grad, sample * width + feature)]), scale, value);
            cursor += 1usize;
        }
        output[p] = value;
    }
}

#[ruda(launch)]
fn finish<S: Float + RudaElement, F: Float + RudaElement, I: Int + RudaElement>(
    indices: &Tensor<I>, order: &Tensor<u32>, prefix: &Tensor<u32>, partial: &Tensor<F>, output: &mut Tensor<S>, padding: i64,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < output.len() {
        let width = output.shape(1usize); let row = i64::cast_from(p / width); let feature = p % width;
        let mut value = F::new(0.0);
        if row != padding {
            let first = bound(indices, order, row, false); let end = bound(indices, order, row, true);
            if first < end {
                let mut chunk = (prefix[first] - 1u32) as usize; let last = prefix[end - 1usize] as usize;
                while chunk < last { value += partial[chunk * width + feature]; chunk += 1usize; }
            }
        }
        output[address(output, p)] = S::cast_from(value);
    }
}

fn grid(blocks: usize) -> RudaCount {
    assert!(blocks > 0 && blocks <= u32::MAX as usize, "embedding launch exceeds U32 indexing");
    let x = blocks.min(131072);
    RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1)
}

fn scratch(c: &ComputeClient<CudaRuntime>, length: usize, dtype: u32) -> View {
    assert!(length <= u32::MAX as usize, "embedding workspace exceeds U32 indexing");
    View::packed(c.empty(length.checked_mul(element_bytes(dtype)).expect("embedding workspace overflow")), vec![length], dtype)
}

fn read_scalar(c: &ComputeClient<CudaRuntime>, value: &View) -> u32 {
    sync(c);
    let bytes = c.read_one(value.handle.clone()).expect("embedding metadata readback failed");
    DOWNLOAD.fetch_add(4, Ordering::Relaxed); assert_eq!(bytes.len(), 4);
    u32::from_le_bytes(bytes[..4].try_into().unwrap())
}

pub(super) fn validate_indices(c: &ComputeClient<CudaRuntime>, indices: &View, rows: usize) {
    if indices.len == 0 { return; }
    if std::env::var("RUDA_TORCH_INDEX_CHECK").as_deref() == Ok("trusted") {
        TRUSTED_INDEX_CALLS.fetch_add(1, Ordering::Relaxed);
        return;
    }
    let status = scratch(c, 1, 5);
    unsafe { clear_status::launch::<CudaRuntime>(c, RudaCount::Static(1, 1, 1), RudaDim::new_1d(1), status.arg()); }
    macro_rules! check { ($i:ty) => { unsafe { check_indices::launch::<$i, CudaRuntime>(c, grid(indices.len.div_ceil(256)), RudaDim::new_1d(256), indices.arg(), status.arg(), rows); }}; }
    if indices.dtype == 4 { check!(i64); } else { check!(i32); }
    LAUNCHES.fetch_add(2, Ordering::Relaxed);
    assert_eq!(read_scalar(c, &status), 0, "RUDA index out of bounds");
}

fn part(value: &View, imaginary: bool) -> View {
    if matches!(value.dtype, 10 | 11 | 14) {
        let result = scientific::component(value, imaginary);
        scientific::indexable(&result); result
    } else { View { handle: value.handle.clone(), shape: value.shape.clone(), strides: value.strides.clone(), len: value.len, dtype: value.dtype } }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64]) {
    assert!(matches!(op, 468 | 469)); assert_eq!(xs.len(), 3);
    let input = &xs[0]; let indices = &xs[1]; let output = &xs[2];
    assert!(matches!(indices.dtype, 4 | 5)); assert_eq!(input.dtype, output.dtype);
    let c = client(); let block = RudaDim::new_1d(256);
    if op == 468 {
        assert!(ints.is_empty()); assert_eq!(input.shape.len(), 2); assert!(input.dtype <= 14);
        assert_eq!(output.len, indices.len.checked_mul(input.shape[1]).expect("embedding output shape overflow"));
        validate_indices(&c, indices, input.shape[0]);
        if output.len == 0 { return; }
        let re = part(input, false); let im = part(input, true); let out_re = part(output, false); let out_im = part(output, true);
        let complex = matches!(input.dtype, 10 | 11 | 14);
        macro_rules! launch { ($s:ty, $i:ty) => { unsafe { gather::launch::<$s, $i, CudaRuntime>(&c, grid(output.len.div_ceil(256)), block,
            re.arg(), im.arg(), indices.arg(), out_re.arg(), out_im.arg(), complex); }}; }
        macro_rules! data { ($s:ty) => { if indices.dtype == 4 { launch!($s, i64); } else { launch!($s, i32); } }; }
        match re.dtype {
            0 => data!(f32), 1 => data!(f16), 2 => data!(bf16), 3 | 8 | 12 | 13 => data!(u8),
            4 => data!(i64), 5 => data!(i32), 6 => data!(i16), 7 => data!(i8), 9 => data!(f64), _ => unreachable!(),
        }
        LAUNCHES.fetch_add(1, Ordering::Relaxed); finish_dispatch(&c); return;
    }
    assert_eq!(ints.len(), 2); assert!(matches!(ints[1], 0 | 1));
    assert!(matches!(input.dtype, 0 | 1 | 2 | 9)); assert_eq!(output.shape.len(), 2);
    let width = output.shape[1]; let padding = ints[0]; let frequency = ints[1] != 0;
    assert_eq!(input.len, indices.len.checked_mul(width).expect("embedding gradient shape overflow"));
    validate_indices(&c, indices, output.shape[0]);
    if output.len == 0 { return; }
    if indices.len == 0 || (indices.len <= 3072 && !frequency) {
        macro_rules! initialize { ($s:ty) => { unsafe { zero::launch::<$s, CudaRuntime>(&c, grid(output.len.div_ceil(256)), block, output.arg()); }}; }
        match input.dtype { 0 => initialize!(f32), 1 => initialize!(f16), 2 => initialize!(bf16), 9 => initialize!(f64), _ => unreachable!() }
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
        if indices.len != 0 {
            macro_rules! launch { ($s:ty, $f:ty, $i:ty) => { unsafe { small_backward::launch::<$s, $f, $i, CudaRuntime>(&c,
                grid(width.div_ceil(32)), RudaDim::new_1d(1024), input.arg(), indices.arg(), output.arg(), padding); }}; }
            macro_rules! data { ($s:ty, $f:ty) => { if indices.dtype == 4 { launch!($s, $f, i64); } else { launch!($s, $f, i32); } }; }
            match input.dtype { 0 => data!(f32, f32), 1 => data!(f16, f32), 2 => data!(bf16, f32), 9 => data!(f64, f64), _ => unreachable!() }
            LAUNCHES.fetch_add(1, Ordering::Relaxed);
        }
        finish_dispatch(&c); return;
    }
    let order = scratch(&c, indices.len, 5);
    let (_sort_workspace, sorts) = unique_ops::stable_order(&c, indices, &order, 1, true);
    let prefix = scratch(&c, indices.len, 5); let size = scratch(&c, 1, 5);
    macro_rules! flags { ($i:ty) => { unsafe { chunk_flags::launch::<$i, CudaRuntime>(&c, grid(indices.len.div_ceil(256)), block, indices.arg(), order.arg(), prefix.arg()); }}; }
    if indices.dtype == 4 { flags!(i64); } else { flags!(i32); }
    let (_levels, scans) = unique_ops::inclusive_prefix(&c, &prefix);
    unsafe { chunk_count::launch::<CudaRuntime>(&c, RudaCount::Static(1, 1, 1), RudaDim::new_1d(1), prefix.arg(), size.arg()); }
    LAUNCHES.fetch_add(sorts + scans + 2, Ordering::Relaxed);
    let chunks = read_scalar(&c, &size) as usize;
    assert!(chunks > 0 && chunks <= indices.len);
    let starts = scratch(&c, chunks, 5); let counts = scratch(&c, if frequency { chunks } else { 1 }, 5);
    let partial = scratch(&c, chunks.checked_mul(width).expect("embedding partial shape overflow"), if input.dtype == 9 { 9 } else { 0 });
    macro_rules! metadata { ($i:ty) => { unsafe { chunk_metadata::launch::<$i, CudaRuntime>(&c, grid(indices.len.div_ceil(256)), block,
        indices.arg(), order.arg(), prefix.arg(), starts.arg(), counts.arg(), frequency); }}; }
    if indices.dtype == 4 { metadata!(i64); } else { metadata!(i32); }
    macro_rules! accumulate { ($s:ty, $f:ty) => { unsafe { partials::launch::<$s, $f, CudaRuntime>(&c, grid(partial.len.div_ceil(256)), block,
        input.arg(), order.arg(), starts.arg(), counts.arg(), partial.arg(), width, frequency); }}; }
    match input.dtype { 0 => accumulate!(f32, f32), 1 => accumulate!(f16, f32), 2 => accumulate!(bf16, f32), 9 => accumulate!(f64, f64), _ => unreachable!() }
    macro_rules! emit { ($s:ty, $f:ty, $i:ty) => { unsafe { finish::launch::<$s, $f, $i, CudaRuntime>(&c, grid(output.len.div_ceil(256)), block,
        indices.arg(), order.arg(), prefix.arg(), partial.arg(), output.arg(), padding); }}; }
    macro_rules! data { ($s:ty, $f:ty) => { if indices.dtype == 4 { emit!($s, $f, i64); } else { emit!($s, $f, i32); } }; }
    match input.dtype { 0 => data!(f32, f32), 1 => data!(f16, f32), 2 => data!(bf16, f32), 9 => data!(f64, f64), _ => unreachable!() }
    LAUNCHES.fetch_add(3, Ordering::Relaxed); finish_dispatch(&c);
}
