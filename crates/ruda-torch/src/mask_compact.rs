use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn flags<T: Numeric + RudaElement>(
    input: &Tensor<T>, imaginary: &Tensor<T>, prefix: &mut Tensor<u32>,
    #[comptime] complex: bool, #[comptime] fp8: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < prefix.len() {
        let value = input[address(input, p)];
        let mut nonzero = value != T::cast_from(0u32);
        if comptime!(fp8) { nonzero = (u8::cast_from(value) & 127u8) != 0u8; }
        if comptime!(complex) { nonzero = nonzero || imaginary[address(imaginary, p)] != T::cast_from(0u32); }
        prefix[p] = u32::cast_from(nonzero);
    }
}

#[ruda(launch)]
fn total(prefix: &Tensor<u32>, count: &mut Tensor<i64>) {
    if UNIT_POS == 0u32 { count[0] = i64::cast_from(prefix[prefix.len() - 1usize]); }
}

#[ruda]
fn selected(prefix: &Tensor<u32>, p: usize) -> bool {
    let mut previous = 0u32;
    if p > 0usize { previous = prefix[p - 1usize]; }
    prefix[p] != previous
}

#[ruda(launch)]
fn positions(
    input: &Tensor<u8>, prefix: &Tensor<u32>, output: &mut Tensor<i64>, rank: usize,
    #[comptime] flat: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < prefix.len() {
        if selected(prefix, p) {
            let row = (prefix[p] - 1u32) as usize;
            if comptime!(flat) { output[address(output, row)] = i64::cast_from(p); }
            else {
                let mut coordinate = p;
                let mut axis = rank;
                while axis > 0usize {
                    axis -= 1usize;
                    output[address(output, row * rank + axis)] = i64::cast_from(coordinate % input.shape(axis));
                    coordinate /= input.shape(axis);
                }
            }
        }
    }
}

#[ruda(launch)]
fn select<T: Numeric + RudaElement>(
    input: &Tensor<T>, imaginary: &Tensor<T>, prefix: &Tensor<u32>,
    output: &mut Tensor<T>, output_imaginary: &mut Tensor<T>, count: usize,
    #[comptime] complex: bool, #[comptime] padded: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < prefix.len() {
        if selected(prefix, p) {
            let target = (prefix[p] - 1u32) as usize;
            output[address(output, target)] = input[address(input, p)];
            if comptime!(complex) {
                output_imaginary[address(output_imaginary, target)] = imaginary[address(imaginary, p)];
            }
        }
    }
    if comptime!(padded) {
        if p >= count && p < output.len() {
            output[address(output, p)] = T::cast_from(0u32);
            if comptime!(complex) { output_imaginary[address(output_imaginary, p)] = T::cast_from(0u32); }
        }
    }
}

#[ruda(launch)]
fn scatter<T: Numeric + RudaElement>(
    prefix: &Tensor<u32>, source: &Tensor<T>, source_imaginary: &Tensor<T>,
    target: &mut Tensor<T>, target_imaginary: &mut Tensor<T>, #[comptime] complex: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < prefix.len() {
        if selected(prefix, p) {
            let from = (prefix[p] - 1u32) as usize;
            target[address(target, p)] = source[address(source, from)];
            if comptime!(complex) {
                target_imaginary[address(target_imaginary, p)] = source_imaginary[address(source_imaginary, from)];
            }
        }
    }
}

fn part(value: &View, imaginary: bool) -> View {
    if matches!(value.dtype, 10 | 11 | 14) {
        let component = scientific::component(value, imaginary);
        scientific::indexable(&component);
        component
    }
    else { View { handle: value.handle.clone(), shape: value.shape.clone(), strides: value.strides.clone(), len: value.len, dtype: value.dtype } }
}

fn grid(work: usize) -> RudaCount {
    assert!(work > 0 && work <= u32::MAX as usize, "mask compaction exceeds U32 indexing");
    let blocks = work.div_ceil(256);
    let x = blocks.min(131072);
    RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1)
}

fn packed(value: &View, len: usize, dtype: u32) {
    assert_eq!(value.dtype, dtype); assert_eq!(value.shape, vec![len]); assert_eq!(value.strides, vec![1usize]);
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64]) {
    assert!(matches!(op, 463..=466));
    assert_eq!(xs.len(), 3);
    let c = client(); let block = RudaDim::new_1d(256);
    if op == 463 {
        assert!(ints.is_empty());
        let input = &xs[0]; let prefix = &xs[1];
        assert!(input.len > 0 && input.dtype <= 14);
        packed(prefix, input.len, 5); packed(&xs[2], 1, 4);
        let re = part(input, false); let im = part(input, true);
        let complex = matches!(input.dtype, 10 | 11 | 14);
        macro_rules! mark { ($t:ty) => { unsafe { flags::launch::<$t, CudaRuntime>(&c, grid(input.len), block,
            re.arg(), im.arg(), prefix.arg(), complex, matches!(input.dtype, 12 | 13)); }}; }
        match re.dtype {
            0 => mark!(f32), 1 => mark!(f16), 2 => mark!(bf16), 3 | 8 | 12 | 13 => mark!(u8),
            4 => mark!(i64), 5 => mark!(i32), 6 => mark!(i16), 7 => mark!(i8), 9 => mark!(f64), _ => unreachable!(),
        }
        let (_levels, scans) = unique_ops::inclusive_prefix(&c, prefix);
        unsafe { total::launch::<CudaRuntime>(&c, RudaCount::Static(1, 1, 1), RudaDim::new_1d(1), prefix.arg(), xs[2].arg()); }
        LAUNCHES.fetch_add(scans + 2, Ordering::Relaxed);
        finish_dispatch(&c);
    } else if op == 464 {
        assert_eq!(ints.len(), 2);
        assert!(ints[0] >= 0 && matches!(ints[1], 0 | 1));
        let input = &xs[0]; let prefix = &xs[1]; let output = &xs[2];
        let rank = ints[0] as usize; let flat = ints[1] != 0;
        packed(prefix, input.len, 5); assert_eq!(output.dtype, 4);
        if rank == 0 { assert_eq!(input.len, 1); } else { assert_eq!(rank, input.shape.len()); }
        if flat { assert_eq!(output.shape.len(), 1); assert!(output.len <= input.len); }
        else { assert_eq!(output.shape.len(), 2); assert_eq!(output.shape[1], rank); assert!(output.shape[0] <= input.len); }
        if output.len == 0 { return; }
        unsafe { positions::launch::<CudaRuntime>(&c, grid(input.len), block, input.arg(), prefix.arg(), output.arg(), rank, flat); }
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
        finish_dispatch(&c);
    } else if op == 465 {
        assert_eq!(ints.len(), 2); assert!(ints[0] >= 0 && matches!(ints[1], 0 | 1));
        let input = &xs[0]; let prefix = &xs[1]; let output = &xs[2];
        let count = ints[0] as usize; let padded = ints[1] != 0;
        packed(prefix, input.len, 5); assert!(input.dtype <= 14); assert_eq!(output.dtype, input.dtype);
        assert!(count <= input.len && count <= output.len);
        if !padded { assert_eq!(output.len, count); }
        if output.len == 0 { return; }
        let re = part(input, false); let im = part(input, true);
        let out_re = part(output, false); let out_im = part(output, true);
        let work = input.len.max(output.len); let complex = matches!(input.dtype, 10 | 11 | 14);
        macro_rules! gather { ($t:ty) => { unsafe { select::launch::<$t, CudaRuntime>(&c, grid(work), block,
            re.arg(), im.arg(), prefix.arg(), out_re.arg(), out_im.arg(), count, complex, padded); }}; }
        match re.dtype {
            0 => gather!(f32), 1 => gather!(f16), 2 => gather!(bf16), 3 | 8 | 12 | 13 => gather!(u8),
            4 => gather!(i64), 5 => gather!(i32), 6 => gather!(i16), 7 => gather!(i8), 9 => gather!(f64), _ => unreachable!(),
        }
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
        finish_dispatch(&c);
    } else {
        assert_eq!(ints.len(), 1); assert!(ints[0] >= 0);
        let prefix = &xs[0]; let source = &xs[1]; let target = &xs[2]; let count = ints[0] as usize;
        packed(prefix, target.len, 5); assert!(source.dtype <= 14); assert_eq!(target.dtype, source.dtype);
        assert!(count <= target.len && count <= source.len);
        if count == 0 { return; }
        let re = part(source, false); let im = part(source, true);
        let out_re = part(target, false); let out_im = part(target, true);
        let complex = matches!(source.dtype, 10 | 11 | 14);
        macro_rules! write { ($t:ty) => { unsafe { scatter::launch::<$t, CudaRuntime>(&c, grid(target.len), block,
            prefix.arg(), re.arg(), im.arg(), out_re.arg(), out_im.arg(), complex); }}; }
        match re.dtype {
            0 => write!(f32), 1 => write!(f16), 2 => write!(bf16), 3 | 8 | 12 | 13 => write!(u8),
            4 => write!(i64), 5 => write!(i32), 6 => write!(i16), 7 => write!(i8), 9 => write!(f64), _ => unreachable!(),
        }
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
        finish_dispatch(&c);
    }
}
