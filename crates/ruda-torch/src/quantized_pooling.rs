use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn window(i: i64, input: i64, kernel: i64, stride: i64, padding: i64, dilation: i64,
    #[comptime] maximum: bool) -> (i64, i64, i64) {
    let mut first = i * stride - padding;
    let mut last = 0i64;
    let mut count = 0i64;
    if comptime!(maximum) {
        last = i64::min(first + (kernel - 1i64) * dilation + 1i64, input);
        while first < 0i64 { first += dilation; }
    } else {
        last = i64::min(first + kernel, input + padding);
        count = last - first;
        first = i64::max(first, 0i64);
        last = i64::min(last, input);
    }
    (first, last, count)
}

#[ruda]
fn bounded(value: f32, minimum: i64, maximum: i64) -> i64 {
    let mut q = f64::cast_from(value);
    if q < f64::cast_from(minimum) { q = f64::cast_from(minimum); }
    if q > f64::cast_from(maximum) { q = f64::cast_from(maximum); }
    i64::cast_from(q)
}

#[ruda]
fn requantize_average(sum: i32, multiplier: f32, zero: i64, minimum: i64, maximum: i64) -> i64 {
    let scale = 1.0f32 / multiplier;
    let inverse = 1.0f32 / scale;
    let q = (f32::cast_from(sum) * inverse).round() + f32::cast_from(zero);
    bounded(q, minimum, maximum)
}

#[ruda(launch)]
fn maximum<Q: Numeric + RudaElement>(
    input: &Tensor<Q>, output: &mut Tensor<Q>,
    kd: i64, kh: i64, kw: i64, sd: i64, sh: i64, sw: i64,
    pd: i64, ph: i64, pw: i64, dd: i64, dh: i64, dw: i64, minimum: i64,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let id = i64::cast_from(input.shape(2)); let ih = i64::cast_from(input.shape(3)); let iw = i64::cast_from(input.shape(4));
        let od = i64::cast_from(output.shape(2)); let oh = i64::cast_from(output.shape(3)); let ow = i64::cast_from(output.shape(4));
        let pos = i64::cast_from(p); let plane = pos / (od * oh * ow);
        let (d0, d1, _) = window(pos / (oh * ow) % od, id, kd, sd, pd, dd, true);
        let (h0, h1, _) = window(pos / ow % oh, ih, kh, sh, ph, dh, true);
        let (w0, w1, _) = window(pos % ow, iw, kw, sw, pw, dw, true);
        let mut best = minimum; let mut d = d0;
        while d < d1 {
            let mut h = h0;
            while h < h1 {
                let mut w = w0;
                while w < w1 {
                    let logical = ((plane * id + d) * ih + h) * iw + w;
                    let value = i64::cast_from(input[address(input, usize::cast_from(logical))]);
                    if value > best { best = value; }
                    w += dw;
                }
                h += dh;
            }
            d += dd;
        }
        output[address(output, p)] = Q::cast_from(best);
    }
}

#[ruda(launch)]
fn average<Q: Numeric + RudaElement>(
    input: &Tensor<Q>, output: &mut Tensor<Q>,
    kd: i64, kh: i64, kw: i64, sd: i64, sh: i64, sw: i64, pd: i64, ph: i64, pw: i64,
    divisor: i64, zero: i64, scale: f64, minimum: i64, maximum: i64,
    #[comptime] include_pad: bool, #[comptime] channels_last: bool,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let id = i64::cast_from(input.shape(2)); let ih = i64::cast_from(input.shape(3)); let iw = i64::cast_from(input.shape(4));
        let od = i64::cast_from(output.shape(2)); let oh = i64::cast_from(output.shape(3)); let ow = i64::cast_from(output.shape(4));
        let pos = i64::cast_from(p); let plane = pos / (od * oh * ow);
        let (d0, d1, dc) = window(pos / (oh * ow) % od, id, kd, sd, pd, 1i64, false);
        let (h0, h1, hc) = window(pos / ow % oh, ih, kh, sh, ph, 1i64, false);
        let (w0, w1, wc) = window(pos % ow, iw, kw, sw, pw, 1i64, false);
        let size = (d1 - d0) * (h1 - h0) * (w1 - w0);
        let mut divide = size;
        if comptime!(include_pad) { divide = dc * hc * wc; }
        if divisor != 0i64 { divide = divisor; }
        let mut sum = 0u32; let mut d = d0;
        while d < d1 {
            let mut h = h0;
            while h < h1 {
                let mut w = w0;
                while w < w1 {
                    let logical = ((plane * id + d) * ih + h) * iw + w;
                    sum += u32::cast_from(i32::cast_from(input[address(input, usize::cast_from(logical))]));
                    w += 1i64;
                }
                h += 1i64;
            }
            d += 1i64;
        }
        let centered = i32::cast_from(sum - u32::cast_from(size * zero));
        let mut q = 0i64;
        if comptime!(channels_last) {
            let fscale = f32::cast_from(scale);
            let multiplier = (fscale / fscale) / f32::cast_from(i32::cast_from(divide));
            q = requantize_average(centered, multiplier, zero, minimum, maximum);
        } else {
            let multiplier = f32::cast_from((scale / scale) / f64::cast_from(divide));
            let value = (f32::cast_from(centered) * multiplier + f32::cast_from(zero)).round();
            q = bounded(value, minimum, maximum);
        }
        output[address(output, p)] = Q::cast_from(q);
    }
}

#[ruda]
fn adaptive_window(i: i64, input: i64, output: i64) -> (i64, i64) {
    let first = i64::cast_from((f32::cast_from(i * input) / f32::cast_from(output)).floor());
    let last = i64::cast_from((f32::cast_from((i + 1i64) * input) / f32::cast_from(output)).ceil());
    (first, last)
}

#[ruda(launch)]
fn adaptive_average<Q: Numeric + RudaElement>(
    input: &Tensor<Q>, output: &mut Tensor<Q>, zero: i64, scale: f64, minimum: i64, maximum: i64,
    #[comptime] channels_last: bool,
) {
    let p = RUDA_POS_X as usize * 4usize + UNIT_POS as usize / 32usize;
    let lane = UNIT_POS as usize % 32usize;
    if p < output.len() {
        let id = i64::cast_from(input.shape(2)); let ih = i64::cast_from(input.shape(3)); let iw = i64::cast_from(input.shape(4));
        let od = i64::cast_from(output.shape(2)); let oh = i64::cast_from(output.shape(3)); let ow = i64::cast_from(output.shape(4));
        let pos = i64::cast_from(p); let plane = pos / (od * oh * ow);
        let (d0, d1) = adaptive_window(pos / (oh * ow) % od, id, od);
        let (h0, h1) = adaptive_window(pos / ow % oh, ih, oh);
        let (w0, w1) = adaptive_window(pos % ow, iw, ow);
        let kd = d1 - d0; let kh = h1 - h0; let kw = w1 - w0; let size = kd * kh * kw;
        let mut sum64 = 0i64; let mut sum32 = 0u32; let mut at = i64::cast_from(lane);
        while at < size {
            let d = d0 + at / (kh * kw); let h = h0 + at / kw % kh; let w = w0 + at % kw;
            let logical = ((plane * id + d) * ih + h) * iw + w;
            let value = input[address(input, usize::cast_from(logical))];
            if comptime!(channels_last) { sum32 += u32::cast_from(i32::cast_from(value)); }
            else { sum64 += i64::cast_from(value); }
            at += 32i64;
        }
        if comptime!(channels_last) {
            sum32 = plane_sum(sum32);
            if lane == 0usize {
                let centered = i32::cast_from(sum32 - u32::cast_from(size * zero));
                let fscale = f32::cast_from(scale);
                let multiplier = (fscale / fscale) / f32::cast_from(i32::cast_from(size));
                output[address(output, p)] = Q::cast_from(requantize_average(centered, multiplier, zero, minimum, maximum));
            }
        } else {
            sum64 = plane_sum(sum64);
            if lane == 0usize {
                let dr = f32::cast_from(1.0f64 / f64::cast_from(kd));
                let dhr = dr / f32::cast_from(kh);
                let dhwr = dhr / f32::cast_from(kw);
                output[address(output, p)] = Q::cast_from((f32::cast_from(sum64) * dhwr).round());
            }
        }
    }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64], values: &[f64]) {
    assert!((562..=564).contains(&op));
    assert_eq!(xs.len(), 2);
    assert_eq!(ints.len(), if op == 562 { 12 } else if op == 563 { 13 } else { 2 });
    assert_eq!(values.len(), if op == 562 { 0 } else { 1 });
    let input = &xs[0]; let output = &xs[1];
    assert_eq!(input.shape.len(), 5); assert_eq!(output.shape.len(), 5);
    assert_eq!(input.dtype, output.dtype);
    assert_eq!(&input.shape[..2], &output.shape[..2]);
    assert!(input.shape[2..].iter().all(|&n| n > 0));
    let (minimum, maximum): (i64, i64) = match input.dtype {
        5 => (i32::MIN as i64, i32::MAX as i64), 7 => (-128, 127), 8 => (0, 255),
        _ => panic!("quantized pooling requires int8, uint8 or int32 payload"),
    };
    if op != 564 {
        assert!(ints[..6].iter().all(|&n| n > 0));
        assert!(ints[6..9].iter().all(|&n| n >= 0));
        if op == 562 { assert!(ints[9..].iter().all(|&n| n > 0)); }
        else {
            assert!((0..=1).contains(&ints[9])); assert!((0..=1).contains(&ints[12]));
            assert!((minimum..=maximum).contains(&ints[11]));
        }
    } else {
        assert!((minimum..=maximum).contains(&ints[0])); assert!((0..=1).contains(&ints[1]));
    }
    if output.len == 0 { return; }
    let c = client();
    let block = RudaDim::new_1d(if op == 564 { 128 } else { 256 });
    let blocks = output.len.div_ceil(if op == 564 { 4 } else { 256 });
    let grid = RudaCount::Static(u32::try_from(blocks).expect("quantized pooling launch overflow"), 1, 1);
    macro_rules! run { ($q:ty) => { unsafe {
        match op {
            562 => maximum::launch::<$q,CudaRuntime>(&c,grid,block,input.arg(),output.arg(),
                ints[0],ints[1],ints[2],ints[3],ints[4],ints[5],ints[6],ints[7],ints[8],
                ints[9],ints[10],ints[11],minimum),
            563 => average::launch::<$q,CudaRuntime>(&c,grid,block,input.arg(),output.arg(),
                ints[0],ints[1],ints[2],ints[3],ints[4],ints[5],ints[6],ints[7],ints[8],
                ints[10],ints[11],values[0],minimum,maximum,ints[9]!=0,ints[12]!=0),
            564 => adaptive_average::launch::<$q,CudaRuntime>(&c,grid,block,input.arg(),output.arg(),
                ints[0],values[0],minimum,maximum,ints[1]!=0),
            _ => unreachable!(),
        };
    } }; }
    match input.dtype { 5=>run!(i32),7=>run!(i8),8=>run!(u8),_=>unreachable!() }
}
