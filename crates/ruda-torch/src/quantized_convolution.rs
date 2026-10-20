use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn pack(
    source: &Tensor<i8>, target: &mut Tensor<i8>, input_group: usize, output_group: usize,
    kernel_size: usize, output_tiles: usize, inner_tiles: usize,
    #[comptime] transpose: bool, #[comptime] reverse: bool,
) {
    let p = ABSOLUTE_POS as usize;
    if p < target.len() {
        if comptime!(reverse) {
            let spatial = p % kernel_size;
            let mut group = 0usize; let mut ic = 0usize; let mut oc = 0usize;
            if comptime!(transpose) {
                let input_channel = p / (output_group * kernel_size);
                group = input_channel / input_group; ic = input_channel % input_group;
                oc = p / kernel_size % output_group;
            } else {
                let output_channel = p / (input_group * kernel_size);
                group = output_channel / output_group; oc = output_channel % output_group;
                ic = p / kernel_size % input_group;
            }
            let k = spatial * input_group + ic;
            let at = (((group * output_tiles + oc / 16usize) * inner_tiles + k / 16usize) * 16usize + k % 16usize) * 16usize + oc % 16usize;
            target[address(target, p)] = source[at];
        } else {
            let group = p / (output_tiles * inner_tiles * 256usize);
            let oc = p / (inner_tiles * 256usize) % output_tiles * 16usize + p % 16usize;
            let k = p / 256usize % inner_tiles * 16usize + p / 16usize % 16usize;
            let mut value = 0i8;
            if oc < output_group && k < input_group * kernel_size {
                let ic = k % input_group; let spatial = k / input_group;
                let mut at = ((group * output_group + oc) * input_group + ic) * kernel_size + spatial;
                if comptime!(transpose) { at = ((group * input_group + ic) * output_group + oc) * kernel_size + spatial; }
                value = source[address(source, at)];
            }
            target[p] = value;
        }
    }
}

#[ruda(launch)]
fn convolution<Q: Numeric + RudaElement>(
    input: &Tensor<Q>, weight: &Tensor<i8>, scales: &Tensor<f32>, zeros: &Tensor<i32>, bias: &Tensor<f32>,
    accum: &Tensor<Q>, output: &mut Tensor<Q>,
    input_group: usize, output_group: usize, rows: usize, row_tiles: usize,
    output_tiles: usize, inner_tiles: usize, blocks: usize,
    kd: i64, kh: i64, kw: i64, sd: i64, sh: i64, sw: i64,
    pd: i64, ph: i64, pw: i64, dd: i64, dh: i64, dw: i64,
    input_zero: i64, output_zero: i64, accum_zero: i64,
    input_scale: f32, output_scale: f32, accum_scale: f32, minimum: i64, maximum: i64,
    #[comptime] transpose: bool, #[comptime] relu: bool, #[comptime] add: bool,
) {
    let block = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    if block >= blocks { terminate!(); }
    let tid = UNIT_POS as usize; let tx = tid % 16usize; let ty = tid / 16usize;
    let group = block / (row_tiles * output_tiles);
    let output_tile = block % output_tiles;
    let row = block / output_tiles % row_tiles * 16usize + ty;
    let oc = output_tile * 16usize + tx;
    let channel = group * output_group + oc;
    let od = output.shape(2); let oh = output.shape(3); let ow = output.shape(4);
    let spatial_size = od * oh * ow;
    let batch = row / spatial_size; let position = row % spatial_size;
    let oz = i64::cast_from(position / (oh * ow));
    let oy = i64::cast_from(position / ow % oh); let ox = i64::cast_from(position % ow);
    let id = i64::cast_from(input.shape(2)); let ih = i64::cast_from(input.shape(3)); let iw = i64::cast_from(input.shape(4));
    let kernel_size = usize::cast_from(kd * kh * kw); let inner = kernel_size * input_group;
    let mut weight_scale = 1.0f32; let mut weight_zero = 0i32;
    if oc < output_group {
        let mut qindex = 0usize; if scales.len() != 1usize { qindex = channel; }
        weight_scale = scales[address(scales, qindex)]; weight_zero = zeros[address(zeros, qindex)];
    }
    let mut a_tile = SharedMemory::<i32>::new(256usize);
    let mut b_tile = SharedMemory::<i32>::new(256usize);
    let mut sum = 0u32;
    for tile in 0..inner_tiles {
        let ak = tile * 16usize + tx; let bk = tile * 16usize + ty;
        let mut a = 0i32; let mut b = 0i32;
        if row < rows && ak < inner {
            let ic = group * input_group + ak % input_group;
            let kernel_position = i64::cast_from(ak / input_group);
            let kz = kernel_position / (kh * kw); let ky = kernel_position / kw % kh; let kx = kernel_position % kw;
            let mut z = oz * sd - pd + kz * dd;
            let mut y = oy * sh - ph + ky * dh;
            let mut x = ox * sw - pw + kx * dw;
            let mut aligned = true;
            if comptime!(transpose) {
                z = oz + pd - kz * dd; y = oy + ph - ky * dh; x = ox + pw - kx * dw;
                aligned = z >= 0i64 && y >= 0i64 && x >= 0i64 && z % sd == 0i64 && y % sh == 0i64 && x % sw == 0i64;
                z /= sd; y /= sh; x /= sw;
            }
            if aligned && z >= 0i64 && z < id && y >= 0i64 && y < ih && x >= 0i64 && x < iw {
                let logical = ((i64::cast_from(batch * input.shape(1) + ic) * id + z) * ih + y) * iw + x;
                a = i32::cast_from(input[address(input, usize::cast_from(logical))]) - i32::cast_from(input_zero);
            }
        }
        if oc < output_group && bk < inner {
            let at = ((group * output_tiles + output_tile) * inner_tiles + tile) * 256usize + tid;
            b = i32::cast_from(weight[at]) - weight_zero;
        }
        a_tile[tid] = a; b_tile[tid] = b;
        sync_ruda();
        for k in 0..16usize { sum += u32::cast_from(a_tile[ty * 16usize + k] * b_tile[k * 16usize + tx]); }
        sync_ruda();
    }
    if row < rows && oc < output_group {
        let logical = (batch * output.shape(1) + channel) * spatial_size + position;
        let dot = f32::cast_from(i32::cast_from(sum));
        let mut offset = 0.0f32; if bias.len() != 0usize { offset = bias[address(bias, channel)]; }
        let product_scale = input_scale * weight_scale;
        let mut scaled = 0.0f32;
        if comptime!(add) {
            let residual = i64::cast_from(accum[address(accum, logical)]) - accum_zero;
            scaled = (dot * product_scale + offset) / output_scale + f32::cast_from(residual) * (accum_scale / output_scale);
        } else {
            let mut corrected = dot;
            if bias.len() != 0usize { corrected += offset / product_scale; }
            scaled = corrected * (product_scale / output_scale);
        }
        let mut q = f64::cast_from(scaled.round()) + f64::cast_from(output_zero);
        if q < f64::cast_from(minimum) { q = f64::cast_from(minimum); }
        if q > f64::cast_from(maximum) { q = f64::cast_from(maximum); }
        if comptime!(relu) { if q < f64::cast_from(output_zero) { q = f64::cast_from(output_zero); } }
        output[address(output, logical)] = Q::cast_from(q);
    }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64], values: &[f64]) {
    assert!(op == 568 || op == 569);
    let c = client();
    if op == 568 {
        assert_eq!(xs.len(), 2); assert_eq!(ints.len(), 6); assert!(values.is_empty());
        assert!(ints[..4].iter().all(|&x| x > 0));
        assert!((0..=1).contains(&ints[4]) && (0..=1).contains(&ints[5]));
        let (g, ipg, opg, area) = (ints[0] as usize, ints[1] as usize, ints[2] as usize, ints[3] as usize);
        let transpose = ints[4] != 0; let reverse = ints[5] != 0;
        let (plain, packed) = if reverse { (&xs[1], &xs[0]) } else { (&xs[0], &xs[1]) };
        assert_eq!(plain.dtype, 7); assert_eq!(packed.dtype, 7); assert_eq!(plain.shape.len(), 5);
        assert_eq!(plain.shape[0], g * if transpose { ipg } else { opg });
        assert_eq!(plain.shape[1], if transpose { opg } else { ipg });
        assert_eq!(plain.shape[2..].iter().product::<usize>(), area);
        let nt = opg.div_ceil(16); let kt = (area * ipg).div_ceil(16);
        assert_eq!(packed.shape, vec![g, nt, kt, 16, 16]);
        assert_eq!(packed.strides, vec![nt*kt*256, kt*256, 256, 16, 1]);
        if xs[1].len == 0 { return; }
        let grid = RudaCount::Static(u32::try_from(xs[1].len.div_ceil(256)).expect("convolution pack grid overflow"), 1, 1);
        unsafe { pack::launch::<CudaRuntime>(&c,grid,RudaDim::new_1d(256),xs[0].arg(),xs[1].arg(),
            ipg,opg,area,nt,kt,transpose,reverse); }
        return;
    }
    assert_eq!(xs.len(), 7); assert_eq!(ints.len(), 19); assert_eq!(values.len(), 3);
    let (input, weight, scales, zeros, bias, accum, output) = (&xs[0], &xs[1], &xs[2], &xs[3], &xs[4], &xs[5], &xs[6]);
    assert_eq!(input.shape.len(), 5); assert_eq!(output.shape.len(), 5);
    assert_eq!(input.shape[0], output.shape[0]); assert_eq!(input.dtype, output.dtype);
    assert_eq!(weight.dtype, 7); assert_eq!(scales.dtype, 0); assert_eq!(zeros.dtype, 5); assert_eq!(bias.dtype, 0);
    assert_eq!(accum.dtype, input.dtype);
    assert!(ints[0] > 0); let g = ints[0] as usize;
    assert!(input.shape[1] > 0 && input.shape[1] % g == 0);
    assert!(output.shape[1] > 0 && output.shape[1] % g == 0);
    assert!(ints[4..7].iter().all(|&x| x == 0 || x == 1));
    assert!(ints[7..13].iter().all(|&x| x > 0));
    assert!(ints[13..16].iter().all(|&x| x >= 0));
    assert!(ints[16..19].iter().all(|&x| x > 0));
    assert!(scales.len == 1 || scales.len == output.shape[1]); assert_eq!(zeros.len, scales.len);
    assert!(bias.len == 0 || bias.len == output.shape[1]);
    if ints[6] != 0 { assert_eq!(accum.shape, output.shape); assert_eq!(ints[4], 0); }
    else { assert_eq!(accum.len, 0); }
    let (minimum, maximum) = match input.dtype { 7 => (-128,127), 8 => (0,255), _ => panic!("quantized convolution requires8-bit input") };
    assert!((minimum..=maximum).contains(&ints[1]) && (minimum..=maximum).contains(&ints[2]));
    if ints[6] != 0 { assert!((minimum..=maximum).contains(&ints[3])); }
    let ipg = input.shape[1] / g; let opg = output.shape[1] / g;
    let area = (ints[7] as usize).checked_mul(ints[8] as usize).and_then(|x|x.checked_mul(ints[9] as usize)).expect("kernel size overflow");
    let nt = opg.div_ceil(16); let kt = area.checked_mul(ipg).expect("convolution reduction overflow").div_ceil(16);
    assert_eq!(weight.shape, vec![g, nt, kt, 16, 16]);
    assert_eq!(weight.strides, vec![nt*kt*256, kt*256, 256, 16, 1]);
    if output.len == 0 { return; }
    let rows = output.len / output.shape[1]; let mt = rows.div_ceil(16);
    let blocks = g.checked_mul(mt).and_then(|x|x.checked_mul(nt)).expect("convolution launch overflow");
    assert!(blocks <= u32::MAX as usize); let x = blocks.min(131072);
    let grid = RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1);
    macro_rules! run { ($q:ty) => { unsafe {
        convolution::launch::<$q,CudaRuntime>(&c,grid,RudaDim::new_1d(256),input.arg(),weight.arg(),scales.arg(),zeros.arg(),
            bias.arg(),accum.arg(),output.arg(),ipg,opg,rows,mt,nt,kt,blocks,
            ints[7],ints[8],ints[9],ints[10],ints[11],ints[12],ints[13],ints[14],ints[15],ints[16],ints[17],ints[18],
            ints[1],ints[2],ints[3],values[0] as f32,values[1] as f32,values[2] as f32,minimum,maximum,
            ints[4]!=0,ints[5]!=0,ints[6]!=0);
    } }; }
    match input.dtype { 7=>run!(i8),8=>run!(u8),_=>unreachable!() }
}
