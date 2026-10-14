use crate::dsl as kernel_dsl;
use crate::dsl::prelude::*;
use half::f16;
use crate::tensor::{RudaTensor, allocation::empty_device_contiguous_dtype, contiguous::into_contiguous};
use ruda_core::{device::Device, tensor::{DType, Shape}, ir::{ElemType, FloatKind, features::MmaConfig}};
use super::packed::{decode_code, encode_code};

/// Encoded matrices A[M,K] and W[N,K], with independent row blocks and common K blocks.
pub fn scaled_mm<R: Runtime>(a: &RudaTensor<R>, w: &RudaTensor<R>, sa: &RudaTensor<R>, sw: &RudaTensor<R>,
    rows_a: usize, rows_w: usize, block_k: usize, format_a: u32, format_w: u32, strategy: u32) -> RudaTensor<R> {
    let [m, k]: [usize; 2] = a.meta.shape().as_slice().try_into().expect("FP8 activation matrix");
    let [n, kw]: [usize; 2] = w.meta.shape().as_slice().try_into().expect("FP8 weight matrix");
    assert!(k == kw && rows_a > 0 && rows_w > 0 && block_k > 0 && block_k.is_multiple_of(32));
    assert!([rows_a, rows_w, block_k].iter().all(|&size| size <= u32::MAX as usize));
    assert!(matches!(format_a, 18 | 19) && matches!(format_w, 18 | 19) && strategy <= 2);
    assert_eq!(a.dtype, DType::U8); assert_eq!(w.dtype, DType::U8);
    assert_eq!(sa.dtype, DType::F32); assert_eq!(sw.dtype, DType::F32);
    let groups = k.div_ceil(block_k);
    assert_eq!(sa.meta.shape().as_slice(), &[m.div_ceil(rows_a), groups]);
    assert_eq!(sw.meta.shape().as_slice(), &[n.div_ceil(rows_w), groups]);
    for tensor in [a, w, sa, sw] {
        assert!(tensor.is_contiguous() && tensor.qparams.is_none() && tensor.meta.num_elements() <= u32::MAX as usize);
        assert_eq!(tensor.device.to_id(), a.device.to_id());
        assert!(tensor.client.same_execution_queue(&a.client));
    }
    assert!(m.checked_mul(n).is_some_and(|size| size <= u32::MAX as usize));
    let dtype = |format| ElemType::Float(if format == 18 {FloatKind::E4M3} else {FloatKind::E5M2}).into();
    let native = strategy != 2 && a.client.features().matmul.mma.contains(&MmaConfig {
        a_type: dtype(format_a), b_type: dtype(format_w), cd_type: DType::F32.into(), m: 16, n: 8, k: 32 });
    assert!(strategy != 1 || native, "native block FP8 MMA is unavailable");
    let carrier = MmaConfig {a_type: DType::F16.into(), b_type: DType::F16.into(), cd_type: DType::F32.into(), m: 16, n: 16, k: 16};
    let hardware = &a.client.properties().hardware;
    assert!(hardware.plane_size_min == 32 && hardware.plane_size_max == 32);
    assert!(native || (hardware.max_shared_memory_size >= 2048 && a.client.features().matmul.cmma.contains(&carrier)));
    let tile = if native {8} else {16};
    assert!(n.div_ceil(tile) <= hardware.max_ruda_count.0 as usize && m.div_ceil(16) <= hardware.max_ruda_count.1 as usize);
    let output = empty_device_contiguous_dtype(a.client.clone(), a.device.clone(), Shape::new([m, n]), DType::F32);
    if m == 0 || n == 0 {return output;}
    let grid = RudaCount::Static(n.div_ceil(tile) as u32, m.div_ceil(16) as u32, 1);
    if native {
        native_block::launch::<R>(&a.client, grid, RudaDim::new_1d(32), a.clone().into_array_arg(), w.clone().into_array_arg(),
            sa.clone().into_array_arg(), sw.clone().into_array_arg(), output.clone().into_array_arg(),
            m as u32, n as u32, k as u32, rows_a as u32, rows_w as u32, block_k as u32, groups as u32, dtype(format_a), dtype(format_w));
    } else {
        carrier_block::launch::<R>(&a.client, grid, RudaDim::new_1d(32), a.clone().into_array_arg(), w.clone().into_array_arg(),
            sa.clone().into_array_arg(), sw.clone().into_array_arg(), output.clone().into_array_arg(),
            m as u32, n as u32, k as u32, rows_a as u32, rows_w as u32, block_k as u32, groups as u32, format_a, format_w);
    }
    output
}

pub fn prepare_scaled_mm<R: Runtime>(a: &RudaTensor<R>, w: &RudaTensor<R>, sa: &RudaTensor<R>, sw: &RudaTensor<R>,
    output: &RudaTensor<R>, rows_a: usize, rows_w: usize, block_k: usize, format_a: u32, format_w: u32,
    strategy: u32) -> Vec<PreparedKernel<R>> {
    let [m, k]: [usize; 2] = a.meta.shape().as_slice().try_into().expect("FP8 activation matrix");
    let [n, kw]: [usize; 2] = w.meta.shape().as_slice().try_into().expect("FP8 weight matrix");
    assert!(k == kw && rows_a > 0 && rows_w > 0 && block_k > 0 && block_k.is_multiple_of(32));
    assert!([rows_a, rows_w, block_k].iter().all(|&size| size <= u32::MAX as usize));
    assert!(matches!(format_a, 18 | 19) && matches!(format_w, 18 | 19) && strategy <= 2);
    assert_eq!(a.dtype, DType::U8); assert_eq!(w.dtype, DType::U8);
    for tensor in [sa, sw, output] {assert_eq!(tensor.dtype, DType::F32);}
    let groups = k.div_ceil(block_k);
    assert_eq!(sa.meta.shape().as_slice(), &[m.div_ceil(rows_a), groups]);
    assert_eq!(sw.meta.shape().as_slice(), &[n.div_ceil(rows_w), groups]);
    assert_eq!(output.meta.shape().as_slice(), &[m, n]);
    for tensor in [a, w, sa, sw, output] {
        assert!(tensor.is_contiguous() && tensor.qparams.is_none() && tensor.meta.num_elements() <= u32::MAX as usize);
        assert_eq!(tensor.device.to_id(), a.device.to_id());
        assert!(tensor.client.same_execution_queue(&a.client));
    }
    if m == 0 || n == 0 {return Vec::new();}
    let dtype = |format| ElemType::Float(if format == 18 {FloatKind::E4M3} else {FloatKind::E5M2}).into();
    let native = strategy != 2 && a.client.features().matmul.mma.contains(&MmaConfig {
        a_type: dtype(format_a), b_type: dtype(format_w), cd_type: DType::F32.into(), m: 16, n: 8, k: 32 });
    assert!(strategy != 1 || native, "native block FP8 graph MMA is unavailable");
    let carrier = MmaConfig {a_type: DType::F16.into(), b_type: DType::F16.into(), cd_type: DType::F32.into(), m: 16, n: 16, k: 16};
    let hardware = &a.client.properties().hardware;
    assert!(hardware.plane_size_min == 32 && hardware.plane_size_max == 32);
    assert!(native || (hardware.max_shared_memory_size >= 2048 && a.client.features().matmul.cmma.contains(&carrier)));
    let tile = if native {8} else {16};
    assert!(n.div_ceil(tile) <= hardware.max_ruda_count.0 as usize && m.div_ceil(16) <= hardware.max_ruda_count.1 as usize);
    let grid = RudaCount::Static(n.div_ceil(tile) as u32, m.div_ceil(16) as u32, 1);
    vec![unsafe {if native {
        native_block::prepare::<R>(&a.client, grid, RudaDim::new_1d(32), a.clone().into_array_arg(), w.clone().into_array_arg(),
            sa.clone().into_array_arg(), sw.clone().into_array_arg(), output.clone().into_array_arg(),
            m as u32, n as u32, k as u32, rows_a as u32, rows_w as u32, block_k as u32, groups as u32, dtype(format_a), dtype(format_w))
    } else {
        carrier_block::prepare::<R>(&a.client, grid, RudaDim::new_1d(32), a.clone().into_array_arg(), w.clone().into_array_arg(),
            sa.clone().into_array_arg(), sw.clone().into_array_arg(), output.clone().into_array_arg(),
            m as u32, n as u32, k as u32, rows_a as u32, rows_w as u32, block_k as u32, groups as u32, format_a, format_w)
    }}]
}

#[ruda(launch)]
pub fn quantize_blocks<F: Float>(input: &Array<F>, encoded: &mut Array<u8>, scales: &mut Array<f32>,
    rows: u32, cols: u32, groups: u32, #[comptime] block_rows: u32, #[comptime] block_cols: u32,
    maximum: f32, #[comptime] format: u32, #[comptime] power_of_two: bool, #[define(F)] _dtype: StorageType) {
    let row_begin = RUDA_POS_X / groups * block_rows;
    let col_begin = RUDA_POS_X % groups * block_cols;
    let lane = UNIT_POS;
    let mut amax = 0.0f32; let mut invalid: u32 = 0u32;
    let mut offset = lane;
    while offset < block_rows * block_cols {
        let row = row_begin + offset / block_cols; let col = col_begin + offset % block_cols;
        if row < rows && col < cols {
            let value = f32::cast_from(input[(row * cols + col) as usize]).abs();
            if value.to_bits() >= 0x7f800000u32 {invalid = 1u32;} else {amax = f32::max(amax, value);}
        }
        offset += 32;
    }
    amax = plane_max(amax); invalid = plane_max(invalid);
    let mut scale = 1.0f32;
    if invalid == 0u32 && amax > 0.0 {scale = f32::max(amax / maximum, f32::from_bits(0x00800000u32));}
    if comptime!(power_of_two) {
        let bits = scale.to_bits(); let mut exponent = bits & 0x7f800000u32;
        if bits & 0x007fffffu32 != 0u32 {exponent += 0x00800000u32;}
        scale = f32::from_bits(exponent);
    }
    if lane == 0 {scales[RUDA_POS_X as usize] = scale;}
    offset = lane;
    while offset < block_rows * block_cols {
        let row = row_begin + offset / block_cols; let col = col_begin + offset % block_cols;
        if row < rows && col < cols {
            let index = (row * cols + col) as usize;
            let value = f32::cast_from(input[index]) / scale;
            let mut code = encode_code(value, format, true);
            if (value.to_bits() & 0x7fffffffu32) >= 0x7f800000u32 {code = encode_code(value, format, false);}
            encoded[index] = code as u8;
        }
        offset += 32;
    }
}

#[ruda(launch)]
pub fn shard_statistics<F: Float>(input: &Array<F>, statistics: &mut Array<f32>,
    local_rows: u32, local_cols: u32, origin_row: u32, origin_col: u32, groups: u32,
    blocks: u32, #[comptime] block_rows: u32, #[comptime] block_cols: u32,
    #[define(F)] _dtype: StorageType) {
    let block = RUDA_POS_X;
    let row_begin = block / groups * block_rows;
    let col_begin = block % groups * block_cols;
    let row_end = u32::min(row_begin + block_rows, origin_row + local_rows);
    let col_end = u32::min(col_begin + block_cols, origin_col + local_cols);
    let row_start = u32::max(row_begin, origin_row);
    let col_start = u32::max(col_begin, origin_col);
    let mut maximum = 0.0f32;
    let mut invalid: u32 = 0u32;
    if row_end > row_start && col_end > col_start {
        let width = col_end - col_start;
        let count = (row_end - row_start) * width;
        let mut offset = UNIT_POS;
        while offset < count {
            let row = row_start + offset / width - origin_row;
            let col = col_start + offset % width - origin_col;
            let value = f32::cast_from(input[(row * local_cols + col) as usize]).abs();
            if value.to_bits() >= 0x7f800000u32 {invalid = 1u32;}
            else {maximum = f32::max(maximum, value);}
            offset += 32;
        }
    }
    maximum = plane_max(maximum);
    invalid = plane_max(invalid);
    if UNIT_POS == 0 {
        statistics[block as usize] = maximum;
        statistics[(blocks + block) as usize] = f32::cast_from(invalid);
    }
}

#[ruda(launch)]
pub fn quantize_shard<F: Float>(input: &Array<F>, statistics: &Array<f32>,
    encoded: &mut Array<u8>, scales: &mut Array<f32>,
    local_rows: u32, local_cols: u32, origin_row: u32, origin_col: u32,
    global_groups: u32, global_blocks: u32, padded_rows: u32, padded_cols: u32,
    groups: u32, #[comptime] block_rows: u32, #[comptime] block_cols: u32,
    maximum: f32, #[comptime] format: u32, #[comptime] power_of_two: bool,
    #[define(F)] _dtype: StorageType) {
    let block_row = RUDA_POS_X / groups;
    let block_col = RUDA_POS_X % groups;
    let global_block = (origin_row / block_rows + block_row) * global_groups
        + origin_col / block_cols + block_col;
    let observed = statistics[global_block as usize];
    let invalid = statistics[(global_blocks + global_block) as usize];
    let mut scale = 1.0f32;
    if invalid == 0.0 && observed > 0.0 {
        scale = f32::max(observed / maximum, f32::from_bits(0x00800000u32));
    }
    if comptime!(power_of_two) {
        let bits = scale.to_bits();
        let mut exponent = bits & 0x7f800000u32;
        if bits & 0x007fffffu32 != 0u32 {exponent += 0x00800000u32;}
        scale = f32::from_bits(exponent);
    }
    if UNIT_POS == 0 {scales[RUDA_POS_X as usize] = scale;}
    let pad_row = origin_row % block_rows;
    let pad_col = origin_col % block_cols;
    let mut offset = UNIT_POS;
    while offset < block_rows * block_cols {
        let row = block_row * block_rows + offset / block_cols;
        let col = block_col * block_cols + offset % block_cols;
        if row < padded_rows && col < padded_cols {
            let mut code = 0u32;
            if row >= pad_row && col >= pad_col {
                let source_row = row - pad_row;
                let source_col = col - pad_col;
                if source_row < local_rows && source_col < local_cols {
                    let value = f32::cast_from(input[(source_row * local_cols + source_col) as usize]) / scale;
                    code = encode_code(value, format, true);
                    if (value.to_bits() & 0x7fffffffu32) >= 0x7f800000u32 {
                        code = encode_code(value, format, false);
                    }
                }
            }
            encoded[(row * padded_cols + col) as usize] = code as u8;
        }
        offset += 32;
    }
}

pub fn linear<R: Runtime>(input: RudaTensor<R>, weight: &RudaTensor<R>, scales: &RudaTensor<R>,
    block: [usize; 2]) -> RudaTensor<R> {
    let input = into_contiguous(input);
    assert!(matches!(input.dtype, DType::F32 | DType::F16 | DType::BF16));
    let [n, k]: [usize; 2] = weight.meta.shape().as_slice().try_into().expect("FP8 matrix");
    assert!(k > 0 && n > 0 && input.meta.shape().last() == Some(&k));
    let m = input.meta.num_elements() / k;
    let [bn, bk] = block;
    assert!(bn > 0 && bn.is_multiple_of(16) && bk > 0 && bk.is_multiple_of(32));
    assert_eq!(weight.dtype, DType::U8);
    assert_eq!(scales.dtype, DType::F32);
    let groups = k.div_ceil(bk);
    assert_eq!(scales.meta.shape().as_slice(), &[n.div_ceil(bn), groups]);
    for value in [&input, weight, scales] {
        assert!(value.is_contiguous() && value.qparams.is_none() && value.meta.num_elements() <= u32::MAX as usize);
        assert_eq!(value.device.to_id(), input.device.to_id());
        assert!(value.client.same_execution_queue(&input.client));
    }
    assert!(m.checked_mul(n).is_some_and(|v| v <= u32::MAX as usize));
    let fp8 = ElemType::Float(FloatKind::E4M3).into();
    let native = input.client.features().matmul.mma.contains(&MmaConfig {
        a_type: fp8, b_type: fp8, cd_type: ElemType::Float(FloatKind::F32).into(), m: 16, n: 8, k: 32 });
    let carrier = MmaConfig { a_type: DType::F16.into(), b_type: DType::F16.into(),
        cd_type: DType::F32.into(), m: 16, n: 16, k: 16 };
    let hardware = &input.client.properties().hardware;
    assert!(hardware.plane_size_min == 32 && hardware.plane_size_max == 32);
    assert!(native || (hardware.max_shared_memory_size >= 2048 && input.client.features().matmul.cmma.contains(&carrier)));
    let tile = if native { 8 } else { 16 };
    assert!(m.checked_mul(groups).is_some_and(|v| v <= hardware.max_ruda_count.0 as usize) && m.div_ceil(16) <= hardware.max_ruda_count.1 as usize
        && n.div_ceil(tile) <= hardware.max_ruda_count.0 as usize);
    let allocate = |shape, dtype| empty_device_contiguous_dtype(input.client.clone(), input.device.clone(), shape, dtype);
    let out = allocate(Shape::new([m, n]), DType::F32);
    if m == 0 { return out; }
    let encoded = allocate(Shape::new([m, k]), DType::U8);
    let activation_scales = allocate(Shape::new([m, groups]), DType::F32);
    quantize_rows::launch::<R>(&input.client, RudaCount::Static((groups * m) as u32, 1, 1), RudaDim::new_1d(32),
        input.clone().into_array_arg(), encoded.clone().into_array_arg(), activation_scales.clone().into_array_arg(),
        k as u32, groups as u32, bk as u32, input.dtype.into());
    let grid = RudaCount::Static(n.div_ceil(tile) as u32, m.div_ceil(16) as u32, 1);
    if native {
        native_block::launch::<R>(&input.client, grid, RudaDim::new_1d(32), encoded.into_array_arg(),
            weight.clone().into_array_arg(), activation_scales.into_array_arg(), scales.clone().into_array_arg(), out.clone().into_array_arg(),
            m as u32, n as u32, k as u32, 1u32, bn as u32, bk as u32, groups as u32, fp8, fp8);
    } else {
        carrier_block::launch::<R>(&input.client, grid, RudaDim::new_1d(32), encoded.into_array_arg(),
            weight.clone().into_array_arg(), activation_scales.into_array_arg(), scales.clone().into_array_arg(), out.clone().into_array_arg(),
            m as u32, n as u32, k as u32, 1u32, bn as u32, bk as u32, groups as u32, 18u32, 18u32);
    }
    out
}

pub fn dequantize<R: Runtime>(weight: &RudaTensor<R>, scales: &RudaTensor<R>, block: [usize; 2]) -> RudaTensor<R> {
    let [n, k]: [usize; 2] = weight.meta.shape().as_slice().try_into().expect("FP8 matrix");
    let [bn, bk] = block;
    assert!(bn > 0 && bk > 0 && weight.dtype == DType::U8 && scales.dtype == DType::F32);
    assert!(weight.is_contiguous() && scales.is_contiguous() && weight.qparams.is_none() && scales.qparams.is_none());
    assert_eq!(weight.device.to_id(), scales.device.to_id());
    assert!(weight.client.same_execution_queue(&scales.client));
    assert_eq!(scales.meta.shape().as_slice(), &[n.div_ceil(bn), k.div_ceil(bk)]);
    assert!(weight.meta.num_elements() <= u32::MAX as usize);
    let out = empty_device_contiguous_dtype(weight.client.clone(), weight.device.clone(), Shape::new([n, k]), DType::F32);
    if out.meta.num_elements() != 0 {
        unpack::launch::<R>(&weight.client, RudaCount::Static(out.meta.num_elements().div_ceil(128) as u32, 1, 1), RudaDim::new_1d(128),
            weight.clone().into_array_arg(), scales.clone().into_array_arg(), out.clone().into_array_arg(),
            k as u32, bn as u32, bk as u32, k.div_ceil(bk) as u32);
    }
    out
}

#[ruda(launch)]
fn quantize_rows<F: Float>(input: &Array<F>, encoded: &mut Array<u8>, scales: &mut Array<f32>,
    k: u32, groups: u32, #[comptime] block: u32, #[define(F)] _dtype: StorageType) {
    let row = RUDA_POS_X / groups; let group = RUDA_POS_X % groups; let lane = UNIT_POS; let begin = group * block;
    let mut maximum = 0.0f32; let mut invalid: u32 = 0u32;
    #[unroll] for step in 0u32..block / 32u32 {
        let col = begin + lane + step * 32;
        if col < k { let x = f32::cast_from(input[(row * k + col) as usize]).abs();
            if x != x { invalid = 1u32; } else { maximum = f32::max(maximum, x); } }
    }
    maximum = plane_max(maximum); invalid = plane_max(invalid);
    let mut scale = f32::max(maximum, 1e-4) / 448.0;
    if invalid != 0u32 { scale = f32::from_bits(0x7fc00000u32); }
    if lane == 0 { scales[(row * groups + group) as usize] = scale; }
    #[unroll] for step in 0u32..block / 32u32 {
        let col = begin + lane + step * 32;
        if col < k {
            let p = (row * k + col) as usize;
            let value = f32::cast_from(input[p]) / scale;
            encoded[p] = encode_code(value, 18u32, false) as u8;
        }
    }
}

#[ruda(launch)]
fn native_block<A: Float, W: Float>(a: &Array<u8>, w: &Array<u8>, sa: &Array<f32>, sw: &Array<f32>, out: &mut Array<f32>,
    m: u32, n: u32, k: u32, #[comptime] bm: u32, #[comptime] bn: u32, #[comptime] bk: u32, groups: u32,
    #[define(A)] _a_fp8: StorageType, #[define(W)] _w_fp8: StorageType) {
    let def = cmma::MmaDefinition::<A, W, f32>::new(16usize, 8usize, 32usize);
    let va = def.vector_size(cmma::MatrixIdent::A); let vb = def.vector_size(cmma::MatrixIdent::B);
    let vc = def.vector_size(cmma::MatrixIdent::Accumulator);
    let size!(NA) = va; let size!(NB) = vb; let size!(NC) = vc;
    let ca = def.vectors_per_lane(cmma::MatrixIdent::A);
    let cb = def.vectors_per_lane(cmma::MatrixIdent::B);
    let cc = def.vectors_per_lane(cmma::MatrixIdent::Accumulator);
    let mut left = Array::<Vector<A, NA>>::new(ca); let mut right = Array::<Vector<W, NB>>::new(cb);
    let mut sum = Array::<Vector<f32, NC>>::new(cc);
    #[unroll] for i in 0..cc { sum[i] = Vector::cast_from(0.0f32); }
    let row_base = RUDA_POS_Y * 16; let col_base = RUDA_POS_X * 8; let lane = UNIT_POS_PLANE;
    for group in 0..groups {
        let mut acc = Array::<Vector<f32, NC>>::new(cc);
        #[unroll] for i in 0..cc { acc[i] = Vector::cast_from(0.0f32); }
        #[unroll] for tile in 0..bk / 32 {
            let base = group * bk + tile * 32;
            #[unroll] for i in 0..ca {
                let mut reg = Vector::empty();
                #[unroll] for j in 0..va {
                    let (r, c) = def.position_of_nth(lane, (i * va + j) as u32, cmma::MatrixIdent::A);
                    let row = row_base + r; let col = base + c; let mut byte = 0u8;
                    if row < m && col < k { byte = a[(row * k + col) as usize]; }
                    reg[j] = A::reinterpret(byte);
                }
                left[i] = reg;
            }
            #[unroll] for i in 0..cb {
                let mut reg = Vector::empty();
                #[unroll] for j in 0..vb {
                    let (r, c) = def.position_of_nth(lane, (i * vb + j) as u32, cmma::MatrixIdent::B);
                    let row = base + r; let col = col_base + c; let mut byte = 0u8;
                    if row < k && col < n { byte = w[(col * k + row) as usize]; }
                    reg[j] = W::reinterpret(byte);
                }
                right[i] = reg;
            }
            def.execute_inplace(&left, &right, &mut acc);
        }
        #[unroll] for i in 0..cc {
            let partial = acc[i]; let mut total = sum[i];
            #[unroll] for j in 0..vc {
                let (r, c) = def.position_of_nth(lane, (i * vc + j) as u32, cmma::MatrixIdent::Accumulator);
                let row = row_base + r; let col = col_base + c;
                if row < m && col < n { total[j] += partial[j] * sa[((row / bm) * groups + group) as usize] * sw[((col / bn) * groups + group) as usize]; }
            }
            sum[i] = total;
        }
    }
    #[unroll] for i in 0..cc {
        let total = sum[i];
        #[unroll] for j in 0..vc {
            let (r, c) = def.position_of_nth(lane, (i * vc + j) as u32, cmma::MatrixIdent::Accumulator);
            let row = row_base + r; let col = col_base + c;
            if row < m && col < n { out[(row * n + col) as usize] = total[j]; }
        }
    }
}

#[ruda(launch)]
fn carrier_block(a: &Array<u8>, w: &Array<u8>, sa: &Array<f32>, sw: &Array<f32>, out: &mut Array<f32>,
    m: u32, n: u32, k: u32, #[comptime] bm: u32, #[comptime] bn: u32, #[comptime] bk: u32, groups: u32,
    #[comptime] format_a: u32, #[comptime] format_w: u32) {
    let rb = RUDA_POS_Y * 16; let cb = RUDA_POS_X * 16; let lane = UNIT_POS as usize;
    let mut left = SharedMemory::<f16>::new_aligned(256usize, 32usize);
    let mut right = SharedMemory::<f16>::new_aligned(256usize, 32usize);
    let mut result = SharedMemory::<f32>::new_aligned(256usize, 32usize);
    let mut sum = Array::<f32>::new(8usize);
    #[unroll] for i in 0usize..8usize { sum[i] = 0.0; }
    for group in 0..groups {
        let acc = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0f32);
        #[unroll] for tile in 0..bk / 16 {
            let base = group * bk + tile * 16;
            #[unroll] for i in 0usize..8usize {
                let t = lane + i * 32; let r = t as u32 / 16; let c = t as u32 % 16;
                let mut av = 0.0f32; let mut bv = 0.0f32;
                if rb + r < m && base + c < k { av = decode_code(a[((rb + r) * k + base + c) as usize] as u32, format_a); }
                if base + r < k && cb + c < n { bv = decode_code(w[((cb + c) * k + base + r) as usize] as u32, format_w); }
                left[t] = f16::cast_from(av); right[t] = f16::cast_from(bv);
            }
            sync_ruda();
            let aa = cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::A, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor, &left.to_slice(), 16u32);
            let bb = cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor, &right.to_slice(), 16u32);
            cmma::execute::<f16, f16, f32, f32>(&aa, &bb, &acc, &acc); sync_ruda();
        }
        cmma::store(&mut result.to_slice_mut(), &acc, 16, cmma::MatrixLayout::RowMajor); sync_ruda();
        #[unroll] for i in 0usize..8usize {
            let t = lane + i * 32; let row = rb + t as u32 / 16; let col = cb + t as u32 % 16;
            if row < m && col < n { sum[i] += result[t] * sa[((row / bm) * groups + group) as usize] * sw[((col / bn) * groups + group) as usize]; }
        }
        sync_ruda();
    }
    #[unroll] for i in 0usize..8usize {
        let t = lane + i * 32; let row = rb + t as u32 / 16; let col = cb + t as u32 % 16;
        if row < m && col < n { out[(row * n + col) as usize] = sum[i]; }
    }
}

#[ruda(launch)]
fn unpack(weight: &Array<u8>, scales: &Array<f32>, out: &mut Array<f32>,
    k: u32, #[comptime] bn: u32, #[comptime] bk: u32, groups: u32) {
    let i = ABSOLUTE_POS as u32;
    if (i as usize) < out.len() { out[i as usize] = decode_code(weight[i as usize] as u32, 18u32)
        * scales[(((i / k) / bn) * groups + (i % k) / bk) as usize]; }
}
