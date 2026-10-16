use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn position() -> usize {
    (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize
}

#[ruda]
fn compare(a: &Tensor<i64>, ai: usize, b: &Tensor<i64>, bi: usize) -> i32 {
    let mut d = 0usize;
    let mut result = 0i32;
    while d < a.shape(0) && result == 0i32 {
        let x = a[d * a.stride(0) + ai * a.stride(1)];
        let y = b[d * b.stride(0) + bi * b.stride(1)];
        if x < y { result = -1i32; } else if x > y { result = 1i32; }
        d += 1usize;
    }
    result
}

#[ruda(launch)]
fn mark(indices: &Tensor<i64>, order: &mut Tensor<u32>, groups: &mut Tensor<u32>, #[comptime] scalar: bool) {
    let p = position();
    if p < order.len() {
        if comptime!(scalar) { order[p] = u32::cast_from(p); }
        let mut first = p == 0usize;
        if p > 0usize && !scalar { first = compare(indices, order[p - 1usize] as usize, indices, order[p] as usize) != 0i32; }
        groups[p] = u32::cast_from(first);
    }
}

#[ruda(launch)]
fn starts(groups: &Tensor<u32>, offsets: &mut Tensor<u32>, count: &mut Tensor<i64>) {
    let p = position();
    if p < groups.len() {
        let mut first = p == 0usize;
        if p > 0usize { first = groups[p] != groups[p - 1usize]; }
        if first { let group = groups[p] as usize; offsets[group - 1usize] = u32::cast_from(p); }
        if p + 1usize == groups.len() { count[0] = i64::cast_from(groups[p]); }
    }
}

#[ruda(launch)]
fn copy_indices(indices: &Tensor<i64>, order: &Tensor<u32>, offsets: &Tensor<u32>, output: &mut Tensor<i64>) {
    let p = position();
    if p < output.len() {
        let d = p / output.shape(1); let row = p % output.shape(1);
        let source = order[offsets[row] as usize] as usize;
        output[address(output, p)] = indices[d * indices.stride(0) + source * indices.stride(1)];
    }
}

#[ruda(launch)]
fn sum_values<S: Numeric + RudaElement, A: Numeric + RudaElement>(values: &Tensor<S>, order: &Tensor<u32>,
    offsets: &Tensor<u32>, output: &mut Tensor<S>, width: usize, #[comptime] boolean: bool,
) {
    let p = position();
    if p < output.len() {
        let row = p / width; let column = p % width;
        let mut end = order.len();
        if row + 1usize < output.shape(0) { end = offsets[row + 1usize] as usize; }
        let mut j = offsets[row] as usize; let mut sum = A::cast_from(0u32);
        while j < end {
            let value = values[address(values, order[j] as usize * width + column)];
            if comptime!(boolean) { sum = A::cast_from(sum != A::cast_from(0u32) || value != S::cast_from(0u32)); }
            else { sum += A::cast_from(value); }
            j += 1usize;
        }
        output[address(output, p)] = S::cast_from(sum);
    }
}

#[ruda(launch)]
fn nonzero_blocks<S: Numeric + RudaElement>(input: &Tensor<S>, prefix: &mut Tensor<u32>, width: usize,
    #[comptime] tiled: bool,
) {
    let mut row = position();
    if comptime!(tiled) { row = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize; }
    if row < prefix.len() {
        let mut nonzero = 0u32;
        let mut column = 0usize; let mut step = 1usize;
        if comptime!(tiled) { column = UNIT_POS as usize; step = 256usize; }
        while column < width {
            if input[address(input, row * width + column)] != S::cast_from(0u32) { nonzero = 1u32; }
            column += step;
        }
        if comptime!(tiled) {
            let total = plane_sum(nonzero);
            let mut warps = SharedMemory::<u32>::new(8usize);
            if UNIT_POS % 32u32 == 0u32 { warps[UNIT_POS as usize / 32usize] = total; }
            sync_ruda();
            if UNIT_POS == 0u32 {
                let mut sum = 0u32;
                for k in 0..8usize { sum += warps[k]; }
                prefix[row] = u32::cast_from(sum != 0u32);
            }
        } else { prefix[row] = nonzero; }
    }
}

#[ruda(launch)]
fn total(prefix: &Tensor<u32>, count: &mut Tensor<i64>) {
    if UNIT_POS == 0u32 { count[0] = i64::cast_from(prefix[prefix.len() - 1usize]); }
}

#[ruda]
fn selected(prefix: &Tensor<u32>, row: usize) -> bool {
    let mut previous = 0u32;
    if row > 0usize { previous = prefix[row - 1usize]; }
    prefix[row] != previous
}

#[ruda(launch)]
fn dense_indices(input: &Tensor<u8>, prefix: &Tensor<u32>, indices: &mut Tensor<i64>) {
    let row = position();
    if row < prefix.len() {
      if selected(prefix, row) {
        let target = prefix[row] as usize - 1usize;
        let mut coordinate = row; let mut d = indices.shape(0);
        while d > 0usize {
            d -= 1usize;
            indices[d * indices.stride(0) + target * indices.stride(1)] = i64::cast_from(coordinate % input.shape(d));
            coordinate /= input.shape(d);
        }
      }
    }
}

#[ruda(launch)]
fn dense_values(input: &Tensor<u8>, prefix: &Tensor<u32>, output: &mut Tensor<u8>, width: usize) {
    let p = position();
    if p < input.len() {
        let row = p / width;
        if selected(prefix, row) {
            let target = (prefix[row] as usize - 1usize) * width + p % width;
            output[address(output, target)] = input[address(input, p)];
        }
    }
}

#[ruda(launch)]
fn transfer(indices: &Tensor<i64>, values: &mut Tensor<u8>, dense: &mut Tensor<u8>, width: usize,
    #[comptime] gather: bool,
) {
    let p = position();
    if p < values.len() {
        let row = p / width; let mut rem = p % width;
        let sd = indices.shape(0); let mut target = 0usize; let mut valid = true;
        for d in 0..sd {
            let coordinate = indices[d * indices.stride(0) + row * indices.stride(1)];
            if coordinate < 0i64 || coordinate >= i64::cast_from(dense.shape(d)) { valid = false; }
            target += usize::cast_from(coordinate) * dense.stride(d);
        }
        let mut d = dense.rank();
        while d > sd {
            d -= 1usize; target += (rem % dense.shape(d)) * dense.stride(d); rem /= dense.shape(d);
        }
        if comptime!(gather) {
            let mut value = 0u8; if valid { value = dense[target]; }
            values[address(values, p)] = value;
        } else if valid { dense[target] = values[address(values, p)]; }
    }
}

#[ruda(launch)]
fn project<S: Numeric + RudaElement, A: Numeric + RudaElement>(
    source_indices: &Tensor<i64>, source_values: &Tensor<S>, order: &Tensor<u32>,
    target_indices: &Tensor<i64>, target_values: &Tensor<S>, output: &mut Tensor<S>, width: usize,
    #[comptime] projection: bool, #[comptime] accumulate: bool, #[comptime] boolean: bool,
) {
    let p = position();
    if p < output.len() {
        let row = p / width; let column = p % width;
        let mut lo = 0usize; let mut hi = order.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2usize;
            if compare(source_indices, order[mid] as usize, target_indices, row) < 0i32 { lo = mid + 1usize; }
            else { hi = mid; }
        }
        let begin = lo; hi = order.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2usize;
            if compare(source_indices, order[mid] as usize, target_indices, row) <= 0i32 { lo = mid + 1usize; }
            else { hi = mid; }
        }
        let mut end = lo;
        if comptime!(!accumulate) { end = usize::min(end, begin + 1usize); }
        let mut sum = A::cast_from(0u32); let mut j = begin;
        while j < end {
            let mut value = S::cast_from(0u32);
            if comptime!(projection) { value = target_values[address(target_values, p)]; }
            else { value = source_values[address(source_values, order[j] as usize * width + column)]; }
            if comptime!(boolean) { sum = A::cast_from(sum != A::cast_from(0u32) || value != S::cast_from(0u32)); }
            else { sum += A::cast_from(value); }
            j += 1usize;
        }
        output[address(output, p)] = S::cast_from(sum);
    }
}

fn grid(blocks: usize) -> RudaCount {
    assert!(blocks > 0 && blocks <= u32::MAX as usize);
    let x = blocks.min(131072);
    RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1)
}

fn packed(value: &View, length: usize, code: u32) {
    assert_eq!(value.dtype, code); assert_eq!(value.shape, vec![length]); assert_eq!(value.strides, vec![1]);
}

fn indices(value: &View) -> (usize, usize) {
    assert_eq!(value.dtype, 4); assert_eq!(value.shape.len(), 2);
    (value.shape[0], value.shape[1])
}

fn width(values: &View, rows: usize) -> usize {
    assert!(!values.shape.is_empty()); assert_eq!(values.shape[0], rows);
    values.shape[1..].iter().try_fold(1usize, |n, &s| n.checked_mul(s)).expect("COO block size overflow")
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64]) {
    let c = client(); let block = RudaDim::new_1d(256);
    match op {
        502 => {
            assert_eq!(xs.len(), 5); assert!(ints.is_empty());
            let (sd, n) = indices(&xs[0]); assert!(n > 0 && n <= u32::MAX as usize);
            packed(&xs[1], n, 5); packed(&xs[2], n, 5); packed(&xs[3], n, 5); packed(&xs[4], 1, 4);
            let mut launches = 0u64;
            let _temporary = if sd > 0 {
                let transposed = View { handle: xs[0].handle.clone(), shape: vec![n, sd],
                    strides: vec![xs[0].strides[1], xs[0].strides[0]], len: xs[0].len, dtype: 4 };
                let (temporary, count) = crate::unique_ops::stable_order(&c, &transposed, &xs[1], sd, false);
                launches += count; Some(temporary)
            } else { None };
            unsafe { mark::launch::<CudaRuntime>(&c, grid(n.div_ceil(256)), block,
                xs[0].arg(), xs[1].arg(), xs[2].arg(), sd == 0); }
            let (_levels, count) = crate::unique_ops::inclusive_prefix(&c, &xs[2]);
            unsafe { starts::launch::<CudaRuntime>(&c, grid(n.div_ceil(256)), block, xs[2].arg(), xs[3].arg(), xs[4].arg()); }
            LAUNCHES.fetch_add(launches + count + 2, Ordering::Relaxed);
        }
        503 => {
            assert_eq!(xs.len(), 6); assert!(ints.is_empty());
            let (sd, n) = indices(&xs[0]); let (od, m) = indices(&xs[4]);
            assert_eq!(sd, od); assert!(m <= n); packed(&xs[2], n, 5); packed(&xs[3], n, 5);
            let w = width(&xs[1], n); assert_eq!(w, width(&xs[5], m));
            assert_eq!(xs[1].shape[1..], xs[5].shape[1..]); assert_eq!(xs[1].dtype, xs[5].dtype);
            if xs[4].len > 0 { unsafe { copy_indices::launch::<CudaRuntime>(&c, grid(xs[4].len.div_ceil(256)), block,
                xs[0].arg(), xs[2].arg(), xs[3].arg(), xs[4].arg()); } LAUNCHES.fetch_add(1, Ordering::Relaxed); }
            if xs[5].len > 0 {
                macro_rules! sum { ($s:ty,$a:ty) => { unsafe { sum_values::launch::<$s,$a,CudaRuntime>(&c,
                    grid(xs[5].len.div_ceil(256)), block, xs[1].arg(), xs[2].arg(), xs[3].arg(), xs[5].arg(), w, xs[1].dtype == 3); }}; }
                match xs[1].dtype { 0 => sum!(f32,f32), 1 => sum!(f16,f32), 2 => sum!(bf16,f32),
                    3 | 8 => sum!(u8,u64), 4 => sum!(i64,u64), 5 => sum!(i32,u64), 6 => sum!(i16,u64), 7 => sum!(i8,u64),
                    9 => sum!(f64,f64), _ => panic!("unsupported COO values dtype") }
                LAUNCHES.fetch_add(1, Ordering::Relaxed);
            }
        }
        504 => {
            assert_eq!(xs.len(), 3); assert_eq!(ints.len(), 1); assert!(ints[0] >= 0);
            let sd = ints[0] as usize; assert!(sd <= xs[0].shape.len());
            let rows: usize = xs[0].shape[..sd].iter().product(); assert!(rows > 0);
            let w: usize = xs[0].shape[sd..].iter().product();
            packed(&xs[1], rows, 5); packed(&xs[2], 1, 4);
            let tiled = w >= 32;
            macro_rules! flags { ($s:ty) => { unsafe { nonzero_blocks::launch::<$s,CudaRuntime>(&c,
                grid(if tiled {rows} else {rows.div_ceil(256)}), block, xs[0].arg(), xs[1].arg(), w, tiled); }}; }
            match xs[0].dtype { 0 => flags!(f32), 1 => flags!(f16), 2 => flags!(bf16), 3 | 8 => flags!(u8),
                4 => flags!(i64), 5 => flags!(i32), 6 => flags!(i16), 7 => flags!(i8), 9 => flags!(f64),
                _ => panic!("unsupported dense COO input dtype") }
            let (_levels, count) = crate::unique_ops::inclusive_prefix(&c, &xs[1]);
            unsafe { total::launch::<CudaRuntime>(&c, RudaCount::Static(1,1,1), block, xs[1].arg(), xs[2].arg()); }
            LAUNCHES.fetch_add(count + 2, Ordering::Relaxed);
        }
        505 => {
            assert_eq!(xs.len(), 4); assert!(ints.is_empty());
            let (sd, n) = indices(&xs[2]); assert!(sd <= xs[0].shape.len());
            let rows: usize = xs[0].shape[..sd].iter().product(); packed(&xs[1], rows, 5);
            assert_eq!(xs[0].dtype, xs[3].dtype); assert_eq!(xs[0].shape[sd..], xs[3].shape[1..]);
            let w = width(&xs[3], n) * element_bytes(xs[0].dtype);
            scientific::indexable(&xs[0].byte_view()); scientific::indexable(&xs[3].byte_view());
            if rows > 0 && sd > 0 { unsafe { dense_indices::launch::<CudaRuntime>(&c, grid(rows.div_ceil(256)), block,
                xs[0].byte_view().arg(), xs[1].arg(), xs[2].arg()); } LAUNCHES.fetch_add(1, Ordering::Relaxed); }
            if xs[0].len > 0 { unsafe { dense_values::launch::<CudaRuntime>(&c, grid(xs[0].byte_view().len.div_ceil(256)), block,
                xs[0].byte_view().arg(), xs[1].arg(), xs[3].byte_view().arg(), w); } LAUNCHES.fetch_add(1, Ordering::Relaxed); }
        }
        506 => {
            assert_eq!(xs.len(), 3); assert_eq!(ints.len(), 1); assert!(matches!(ints[0],0 | 1));
            let (sd, n) = indices(&xs[0]); assert!(sd <= xs[2].shape.len());
            let w = width(&xs[1], n) * element_bytes(xs[1].dtype);
            assert_eq!(xs[1].dtype, xs[2].dtype); assert_eq!(xs[1].shape[1..], xs[2].shape[sd..]);
            let bytes = xs[1].byte_view();
            scientific::indexable(&bytes); scientific::indexable(&xs[2].byte_view());
            if bytes.len > 0 { unsafe { transfer::launch::<CudaRuntime>(&c, grid(bytes.len.div_ceil(256)), block,
                xs[0].arg(), bytes.arg(), xs[2].byte_view().arg(), w, ints[0] == 1); } LAUNCHES.fetch_add(1, Ordering::Relaxed); }
        }
        507 => {
            assert_eq!(xs.len(), 6); assert_eq!(ints.len(), 2); assert!(ints.iter().all(|&x| matches!(x,0 | 1)));
            let (sd, n) = indices(&xs[0]); let (td, m) = indices(&xs[3]); assert_eq!(sd, td);
            packed(&xs[2], n, 5); let w = width(&xs[5], m);
            assert_eq!(xs[1].dtype, xs[5].dtype); assert_eq!(xs[4].dtype, xs[5].dtype);
            if ints[0] == 0 { assert_eq!(width(&xs[1], n), w); assert_eq!(xs[1].shape[1..], xs[5].shape[1..]); }
            else { assert_eq!(xs[4].shape, xs[5].shape); }
            if xs[5].len > 0 {
                macro_rules! select { ($s:ty,$a:ty) => { unsafe { project::launch::<$s,$a,CudaRuntime>(&c,
                    grid(xs[5].len.div_ceil(256)), block, xs[0].arg(), xs[1].arg(), xs[2].arg(), xs[3].arg(),
                    xs[4].arg(), xs[5].arg(), w, ints[0] != 0, ints[1] != 0, xs[5].dtype == 3); }}; }
                match xs[5].dtype { 0 => select!(f32,f32), 1 => select!(f16,f32), 2 => select!(bf16,f32), 3 | 8 => select!(u8,u64),
                    4 => select!(i64,u64), 5 => select!(i32,u64), 6 => select!(i16,u64), 7 => select!(i8,u64), 9 => select!(f64,f64),
                    _ => panic!("unsupported COO projection dtype") }
                LAUNCHES.fetch_add(1, Ordering::Relaxed);
            }
        }
        _ => unreachable!(),
    }
    finish_dispatch(&c);
}
