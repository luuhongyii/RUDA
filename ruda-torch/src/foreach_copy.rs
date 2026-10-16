use super::*;
use ruda_kernel::dsl::prelude::Tensor;
use ruda_kernel::quantization::packed::decode_code;

#[ruda]
pub(super) fn encode_fp8(value: f32, #[comptime] code: u32) -> u8 {
    let raw = u32::reinterpret(value);
    if comptime!(code == 15) {
        let mut exponent = (raw >> 23u32) & 255u32;
        if exponent != 255u32 && (raw & 0x400000u32) != 0u32 && ((raw & 0x3fffffu32) != 0u32 || exponent != 0u32) {
            exponent += 1u32;
        }
        return u8::cast_from(exponent);
    }
    let sign = raw & 0x80000000u32;
    let magnitude = raw ^ sign;
    let mut result = 0u32;
    if comptime!(code == 12) {
        if magnitude >= (1087u32 << 20u32) {
            result = 0x7eu32;
            if magnitude > 0x7f800000u32 { result = 0x7fu32; }
        } else if magnitude < (121u32 << 23u32) {
            let mask = 141u32 << 23u32;
            result = u32::reinterpret(f32::reinterpret(magnitude) + f32::reinterpret(mask)) - mask;
        } else {
            let odd = (magnitude >> 20u32) & 1u32;
            let rounded = magnitude + (u32::cast_from(-120i32) << 23u32) + 0x7ffffu32 + odd;
            result = (rounded >> 20u32) & 0xffu32;
            if result == 0x7fu32 { result = 0x7eu32; }
        }
    } else {
        if magnitude >= (143u32 << 23u32) {
            result = 0x7cu32;
            if magnitude > 0x7f800000u32 { result = 0x7fu32; }
        } else if magnitude < (113u32 << 23u32) {
            let mask = 134u32 << 23u32;
            result = u32::reinterpret(f32::reinterpret(magnitude) + f32::reinterpret(mask)) - mask;
        } else {
            let odd = (magnitude >> 21u32) & 1u32;
            let rounded = magnitude + (u32::cast_from(-112i32) << 23u32) + 0xfffffu32 + odd;
            result = (rounded >> 21u32) & 0xffu32;
        }
    }
    u8::cast_from(result | (sign >> 24u32))
}

#[ruda]
pub(super) fn decode_fp8(value: u32, #[comptime] code: u32) -> f32 {
    if comptime!(code == 15) {
        let mut bits = value << 23u32;
        if value == 0u32 { bits = 0x00400000u32; }
        if value == 255u32 { bits = 0x7f800001u32; }
        f32::reinterpret(bits)
    } else { decode_code(value, comptime!(if code == 12 {18u32} else {19u32})) }
}

#[ruda]
fn cast<I: Numeric, O: Numeric>(value: I, #[comptime] source: u32, #[comptime] destination: u32) -> O {
    let mut result = O::cast_from(0u32);
    if comptime!(source == 12 || source == 13 || source == 15) {
        let decoded = decode_fp8(u32::cast_from(value), source);
        if comptime!(destination == 12 || destination == 13 || destination == 15) { result = O::cast_from(encode_fp8(decoded, destination)); }
        else if comptime!(destination == 3) { result = O::cast_from(decoded != 0.0f32); }
        else { result = O::cast_from(decoded); }
    } else if comptime!(destination == 12 || destination == 13 || destination == 15) {
        result = O::cast_from(encode_fp8(f32::cast_from(value), destination));
    } else if comptime!(destination == 3) {
        result = O::cast_from(value != I::cast_from(0u32));
    } else if comptime!(source == 1 || source == 2 || destination == 1 || destination == 2) {
        result = O::cast_from(f32::cast_from(value));
    } else { result = O::cast_from(value); }
    result
}

#[ruda(launch)]
fn copy<I: Numeric + RudaElement, O: Numeric + RudaElement>(
    inputs: &Sequence<Tensor<I>>, outputs: &mut Sequence<Tensor<O>>,
    #[comptime] source: u32, #[comptime] destination: u32,
) {
    let input_width = comptime!(if source == 10 || source == 11 { 2usize } else { 1usize });
    let output_width = comptime!(if destination == 10 || destination == 11 { 2usize } else { 1usize });
    let block = RUDA_POS_X as usize;
    let mut first = 0usize;
    #[unroll]
    for tensor in 0..inputs.len() {
        let input = inputs.index(tensor);
        let len = input.len() / input_width;
        let blocks = len / 65536usize + usize::cast_from(len % 65536usize != 0);
        if block >= first && block < first + blocks {
            let output = outputs.index_mut(tensor);
            let base = (block - first) * 65536usize;
            let remaining = len - base;
            let mut offset = UNIT_POS as usize;
            while offset < remaining && offset < 65536usize {
                let p = base + offset;
                let real = input[p * input_width];
                let mut imag = I::cast_from(0u32);
                if comptime!(input_width == 2 && output_width == 2) { imag = input[p * input_width + 1usize]; }
                let real_out = cast::<I, O>(real, source, destination);
                let mut imag_out = O::cast_from(0u32);
                if comptime!(output_width == 2 && input_width == 2) { imag_out = cast::<I, O>(imag, source, destination); }
                output[p * output_width] = real_out;
                if comptime!(output_width == 2) { output[p * output_width + 1usize] = imag_out; }
                offset += 512usize;
            }
        }
        first += blocks;
    }
}

pub(super) fn dispatch(xs: &[View], ints: &[i64]) {
    assert_eq!(ints.len(), 1);
    assert!(ints[0] > 0);
    let n = usize::try_from(ints[0]).expect("foreach copy tensor count overflow");
    assert_eq!(xs.len(), n.checked_mul(2).expect("foreach copy tensor count overflow"));
    let source = xs[0].dtype;
    let destination = xs[n].dtype;
    assert!((source <= 13 || source == 15) && (destination <= 13 || destination == 15));
    for i in 0..n {
        assert_eq!(xs[i].dtype, source);
        assert_eq!(xs[n + i].dtype, destination);
        assert_eq!(xs[i].shape, vec![xs[i].len]);
        assert_eq!(xs[n + i].shape, xs[i].shape);
        assert_eq!(xs[i].strides, vec![1usize]);
        assert_eq!(xs[n + i].strides, vec![1usize]);
    }
    if xs[..n].iter().all(|x| x.len == 0) { return; }
    let storage: Vec<_> = xs.iter().map(|x| {
        let width = if matches!(x.dtype, 10 | 11) { 2 } else { 1 };
        let len = x.len.checked_mul(width).expect("foreach copy component count overflow");
        let dtype = match x.dtype { 10 => 0, 11 => 9, 12 | 13 | 15 => 8, _ => x.dtype };
        let view = View { handle: x.handle.clone(), shape: vec![len], strides: vec![1], len, dtype };
        crate::scientific::indexable(&view);
        view
    }).collect();
    let indices: Vec<_> = (0..n).filter(|&i| xs[i].len != 0).collect();
    let c = client();
    for batch in indices.chunks(16) {
        let blocks: usize = batch.iter().map(|&i| xs[i].len.div_ceil(65536)).sum();
        let grid = RudaCount::Static(u32::try_from(blocks).expect("foreach copy grid overflow"), 1, 1);
        let block = RudaDim::new_1d(512);
        macro_rules! run { ($i:ty, $o:ty) => { unsafe {
            let inputs = batch.iter().map(|&i| storage[i].arg()).collect::<SequenceArg<CudaRuntime, Tensor<$i>>>();
            let outputs = batch.iter().map(|&i| storage[n + i].arg()).collect::<SequenceArg<CudaRuntime, Tensor<$o>>>();
            copy::launch::<$i, $o, CudaRuntime>(&c, grid, block, inputs, outputs, source, destination);
        }}; }
        macro_rules! target { ($i:ty) => { match destination {
            0 | 10 => run!($i, f32), 1 => run!($i, f16), 2 => run!($i, bf16),
            3 | 8 | 12 | 13 | 15 => run!($i, u8), 4 => run!($i, i64), 5 => run!($i, i32),
            6 => run!($i, i16), 7 => run!($i, i8), 9 | 11 => run!($i, f64), _ => unreachable!(),
        }}; }
        match source {
            0 | 10 => target!(f32), 1 => target!(f16), 2 => target!(bf16),
            3 | 8 | 12 | 13 | 15 => target!(u8), 4 => target!(i64), 5 => target!(i32),
            6 => target!(i16), 7 => target!(i8), 9 | 11 => target!(f64), _ => unreachable!(),
        }
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
    }
    finish_dispatch(&c);
}
