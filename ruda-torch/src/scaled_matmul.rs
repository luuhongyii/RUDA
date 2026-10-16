use super::*;
use ruda_core::{ir::{ElemType, FloatKind, features::MmaConfig}, tensor::DType};
use ruda_kernel::{dsl::prelude::Tensor, quantization::packed::{decode_code, encode_code}};

#[ruda]
fn scale<S: Numeric>(x: &Tensor<S>, outer: usize, inner: usize, #[comptime] mode: u32,
    #[comptime] right: bool,
) -> f32 {
    if comptime!(mode == 0) { f32::cast_from(x[0]) }
    else if comptime!(mode == 1) { f32::cast_from(x[crate::scientific::address(x, outer)]) }
    else {
        let mut row = outer;
        if comptime!(mode == 3 || mode == 5) { row /= 128usize; }
        let column = inner / 128usize;
        if comptime!(mode == 5 || (mode < 4 && right)) { f32::cast_from(x[column * x.stride(0) + row * x.stride(1)]) }
        else { f32::cast_from(x[row * x.stride(0) + column * x.stride(1)]) }
    }
}

#[ruda]
pub(super) fn block_index(outer: usize, inner: usize, k: usize, #[comptime] group: usize) -> usize {
    let tile = comptime!(group * 4usize);
    let columns = (k / tile + usize::cast_from(k % tile != 0usize)) * 4usize;
    (outer / 128usize * columns + inner / tile * 4usize) * 128usize
        + outer % 32usize * 16usize + outer % 128usize / 32usize * 4usize + inner / group % 4usize
}

#[ruda]
pub(super) fn scaled_block<S: Numeric>(sa: &Tensor<S>, sb: &Tensor<S>, value: f32, row: usize, column: usize,
    inner: usize, k: usize, #[comptime] mode_a: u32, #[comptime] mode_b: u32,
) -> f32 {
    if comptime!(mode_a == 6) {
        let ea = u32::cast_from(sa[block_index(row, inner, k, 32usize)]);
        let eb = u32::cast_from(sb[block_index(column, inner, k, 32usize)]);
        let mut result = f32::reinterpret(0x7fc00000u32);
        if ea != 255u32 && eb != 255u32 {
            let power = f64::reinterpret(u64::cast_from(ea + eb + 769u32) << 52u32);
            result = f32::cast_from(f64::cast_from(value) * power);
        }
        result
    } else { value * scale(sa, row, inner, mode_a, false) * scale(sb, column, inner, mode_b, true) }
}

#[ruda]
pub(super) fn store<O: Numeric, P: Float>(bias: &Tensor<P>, out: &mut Tensor<O>, row: usize, column: usize,
    value: f32, #[comptime] has_bias: bool, #[comptime] output_code: u32,
) {
    let mut value = value;
    if comptime!(has_bias) { value += f32::cast_from(bias[column]); }
    let converted = if comptime!(output_code == 12 || output_code == 13) {
        O::cast_from(encode_code(value, comptime!(if output_code == 12 {18u32} else {19u32}), true))
    } else { O::cast_from(value) };
    out[row * out.stride(0) + column * out.stride(1)] = converted;
}

#[ruda(launch)]
fn zero<O: Numeric>(out: &mut Tensor<O>, #[define(O)] _output_type: StorageType) {
    let i = ABSOLUTE_POS as usize;
    if i < out.len() { out[crate::scientific::address(out, i)] = O::cast_from(0u32); }
}

#[ruda(launch)]
fn native<A: Float, B: Float, O: Numeric, P: Float, S: Numeric>(
    a: &Tensor<u8>, b: &Tensor<u8>, sa: &Tensor<S>, sb: &Tensor<S>, bias: &Tensor<P>, out: &mut Tensor<O>,
    #[comptime] mode_a: u32, #[comptime] mode_b: u32, #[comptime] fast: bool,
    #[comptime] has_bias: bool, #[comptime] output_code: u32,
    #[define(A)] _a_type: StorageType, #[define(B)] _b_type: StorageType,
    #[define(O)] _output_type: StorageType, #[define(P)] _bias_type: StorageType, #[define(S)] _scale_type: StorageType,
) {
    let m = a.shape(0); let k = a.shape(1); let n = b.shape(1);
    let row_base = RUDA_POS_Y as usize * 16usize; let column_base = RUDA_POS_X as usize * 8usize;
    let lane = UNIT_POS_PLANE;
    let def = cmma::MmaDefinition::<A, B, f32>::new(16usize, 8usize, 32usize);
    let va = def.vector_size(cmma::MatrixIdent::A); let vb = def.vector_size(cmma::MatrixIdent::B);
    let vc = def.vector_size(cmma::MatrixIdent::Accumulator);
    let size!(NA) = va; let size!(NB) = vb; let size!(NC) = vc;
    let ca = def.vectors_per_lane(cmma::MatrixIdent::A); let cb = def.vectors_per_lane(cmma::MatrixIdent::B);
    let cc = def.vectors_per_lane(cmma::MatrixIdent::Accumulator);
    let mut ra = Array::<Vector<A, NA>>::new(ca); let mut rb = Array::<Vector<B, NB>>::new(cb);
    let mut rc = Array::<Vector<f32, NC>>::new(cc); let mut group = Array::<Vector<f32, NC>>::new(cc);
    let mut total = Array::<Vector<f32, NC>>::new(cc);
    #[unroll] for i in 0..cc {
        rc[i] = Vector::cast_from(0.0f32); group[i] = Vector::cast_from(0.0f32); total[i] = Vector::cast_from(0.0f32);
    }
    let mut base = 0usize;
    while base < k {
        #[unroll] for i in 0..ca {
            let mut reg = Vector::empty();
            #[unroll] for j in 0..va {
                let (r, c) = def.position_of_nth(lane, (i * va + j) as u32, cmma::MatrixIdent::A);
                let row = row_base + r as usize; let col = base + c as usize; let mut code = 0u8;
                if row < m && col < k { code = a[row * a.stride(0) + col * a.stride(1)]; }
                reg[j] = A::reinterpret(code);
            }
            ra[i] = reg;
        }
        #[unroll] for i in 0..cb {
            let mut reg = Vector::empty();
            #[unroll] for j in 0..vb {
                let (r, c) = def.position_of_nth(lane, (i * vb + j) as u32, cmma::MatrixIdent::B);
                let row = base + r as usize; let col = column_base + c as usize; let mut code = 0u8;
                if row < k && col < n { code = b[row * b.stride(0) + col * b.stride(1)]; }
                reg[j] = B::reinterpret(code);
            }
            rb[i] = reg;
        }
        def.execute_inplace(&ra, &rb, &mut rc);
        if comptime!(!fast) {
            #[unroll] for i in 0..cc { group[i] = group[i] + rc[i]; rc[i] = Vector::cast_from(0.0f32); }
        }
        let mut end_group = k - base <= 32usize;
        if comptime!(mode_a >= 2) { end_group = end_group || (base + 32usize) % 128usize == 0usize; }
        if comptime!(mode_a == 6) { end_group = true; }
        if end_group {
            #[unroll] for i in 0..cc {
                let mut value = group[i];
                if comptime!(fast) { value = rc[i]; }
                if comptime!(mode_a >= 2) {
                    let mut sum = total[i];
                    #[unroll] for j in 0..vc {
                        let (r, c) = def.position_of_nth(lane, (i * vc + j) as u32, cmma::MatrixIdent::Accumulator);
                        let row = row_base + r as usize; let col = column_base + c as usize;
                        if row < m && col < n { sum[j] += scaled_block(sa, sb, value[j], row, col, base, k, mode_a, mode_b); }
                    }
                    total[i] = sum;
                } else { total[i] = value; }
                rc[i] = Vector::cast_from(0.0f32); group[i] = Vector::cast_from(0.0f32);
            }
        }
        if k - base <= 32usize { base = k; } else { base += 32usize; }
    }
    #[unroll] for i in 0..cc {
        let reg = total[i];
        #[unroll] for j in 0..vc {
            let (r, c) = def.position_of_nth(lane, (i * vc + j) as u32, cmma::MatrixIdent::Accumulator);
            let row = row_base + r as usize; let col = column_base + c as usize;
            if row < m && col < n {
                let mut value = reg[j];
                if comptime!(mode_a < 2) { value = value * scale(sa, row, 0usize, mode_a, false) * scale(sb, col, 0usize, mode_b, true); }
                store(bias, out, row, col, value, has_bias, output_code);
            }
        }
    }
}

#[ruda(launch)]
fn carrier<O: Numeric, P: Float, S: Numeric>(
    a: &Tensor<u8>, b: &Tensor<u8>, sa: &Tensor<S>, sb: &Tensor<S>, bias: &Tensor<P>, out: &mut Tensor<O>,
    #[comptime] mode_a: u32, #[comptime] mode_b: u32, #[comptime] format_a: u32, #[comptime] format_b: u32,
    #[comptime] has_bias: bool, #[comptime] output_code: u32,
    #[define(O)] _output_type: StorageType, #[define(P)] _bias_type: StorageType, #[define(S)] _scale_type: StorageType,
) {
    let m = a.shape(0); let k = a.shape(1); let n = b.shape(1);
    let row_base = RUDA_POS_Y as usize * 16usize; let column_base = RUDA_POS_X as usize * 16usize;
    let lane = UNIT_POS as usize;
    let mut left = SharedMemory::<f16>::new_aligned(256usize, 32usize);
    let mut right = SharedMemory::<f16>::new_aligned(256usize, 32usize);
    let mut partial = SharedMemory::<f32>::new_aligned(256usize, 32usize);
    let mut total = Array::<f32>::new(8usize);
    #[unroll] for i in 0usize..8usize { total[i] = 0.0f32; }
    let acc = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0f32);
    let mut base = 0usize;
    while base < k {
        #[unroll] for i in 0usize..8usize {
            let t = lane + i * 32usize; let row = t / 16usize; let col = t % 16usize;
            let mut av = 0.0f32; let mut bv = 0.0f32;
            if row_base + row < m && base + col < k { av = decode_code(a[(row_base + row) * a.stride(0) + (base + col) * a.stride(1)] as u32, format_a); }
            if base + row < k && column_base + col < n { bv = decode_code(b[(base + row) * b.stride(0) + (column_base + col) * b.stride(1)] as u32, format_b); }
            left[t] = f16::cast_from(av); right[t] = f16::cast_from(bv);
        }
        sync_ruda();
        let aa = cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::A, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor, &left.to_slice(), 16u32);
        let bb = cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor, &right.to_slice(), 16u32);
        cmma::execute::<f16, f16, f32, f32>(&aa, &bb, &acc, &acc); sync_ruda();
        let mut end_group = k - base <= 16usize;
        if comptime!(mode_a >= 2) { end_group = end_group || (base + 16usize) % 128usize == 0usize; }
        if comptime!(mode_a == 6) { end_group = k - base <= 16usize || (base + 16usize) % 32usize == 0usize; }
        if end_group {
            cmma::store(&mut partial.to_slice_mut(), &acc, 16u32, cmma::MatrixLayout::RowMajor); sync_ruda();
            #[unroll] for i in 0usize..8usize {
                let t = lane + i * 32usize; let row = row_base + t / 16usize; let col = column_base + t % 16usize;
                if row < m && col < n {
                    if comptime!(mode_a >= 2) { total[i] += scaled_block(sa, sb, partial[t], row, col, base, k, mode_a, mode_b); }
                    else { total[i] = partial[t]; }
                }
            }
            cmma::fill(&acc, 0.0f32); sync_ruda();
        }
        if k - base <= 16usize { base = k; } else { base += 16usize; }
    }
    #[unroll] for i in 0usize..8usize {
        let t = lane + i * 32usize; let row = row_base + t / 16usize; let col = column_base + t % 16usize;
        if row < m && col < n {
            let mut value = total[i];
            if comptime!(mode_a < 2) { value = value * scale(sa, row, 0usize, mode_a, false) * scale(sb, col, 0usize, mode_b, true); }
            store(bias, out, row, col, value, has_bias, output_code);
        }
    }
}

pub(super) fn dispatch(xs: &[View], ints: &[i64]) {
    assert_eq!(xs.len(), 6); assert_eq!(ints.len(), 4);
    let (a, b, sa, sb, bias, out) = (&xs[0], &xs[1], &xs[2], &xs[3], &xs[4], &xs[5]);
    let (ma, mb, fast, has_bias) = (ints[0], ints[1], ints[2] != 0, ints[3] != 0);
    assert!(matches!((ma, mb), (0, 0) | (1, 1) | (2, 2) | (2, 3) | (3, 2) | (4, 4) | (4, 5) | (5, 4) | (6, 6)));
    assert!(ma < 2 || ma == 6 || !fast); assert!(matches!(ints[2], 0 | 1) && matches!(ints[3], 0 | 1));
    assert!(matches!(a.dtype, 12 | 13) && matches!(b.dtype, 12 | 13)); assert!(a.dtype != 13 || b.dtype != 13);
    if matches!(ma, 4 | 5) { assert_eq!(a.dtype, 12); assert_eq!(b.dtype, 12); }
    assert_eq!(a.shape.len(), 2); assert_eq!(b.shape.len(), 2);
    assert_eq!(sa.dtype, if ma == 6 {15} else {0}); assert_eq!(sb.dtype, sa.dtype);
    let (m, k, n) = (a.shape[0], a.shape[1], b.shape[1]);
    assert_eq!(b.shape[0], k); assert_eq!(out.shape, vec![m, n]); assert!(k % 16 == 0 && n % 16 == 0);
    assert!(matches!(out.dtype, 0 | 1 | 2 | 12 | 13));
    for (s, mode, outer, right) in [(sa, ma, m, false), (sb, mb, n, true)] {
        if mode == 0 { assert_eq!(s.len, 1); }
        else if mode == 6 {
            let count = outer.div_ceil(128).checked_mul(128).and_then(|r| r.checked_mul(k.div_ceil(128) * 4)).expect("MXFP8 scale size overflow");
            assert_eq!(s.len, count);
            let mut stride = 1usize;
            for d in (0..s.shape.len()).rev() { assert!(s.shape[d] <= 1 || s.strides[d] == stride); stride *= s.shape[d]; }
        }
        else if mode == 4 { assert_eq!(s.shape, vec![outer, k.div_ceil(128)]); }
        else if mode == 5 { assert_eq!(s.shape, vec![k.div_ceil(128).div_ceil(4) * 4, outer.div_ceil(128)]); }
        else {
            let rows = if mode == 3 {outer.div_ceil(128)} else {outer};
            let cols = if mode == 1 {1} else {k.div_ceil(128)};
            assert_eq!(s.shape, if right {vec![cols, rows]} else {vec![rows, cols]});
        }
    }
    if has_bias {
        assert!(matches!(bias.dtype, 1 | 2)); assert_eq!(bias.shape, vec![n]); assert_eq!(bias.strides, vec![1]);
        assert!(out.dtype != 0 && (out.dtype != 1 || bias.dtype == 1) && (out.dtype != 2 || bias.dtype == 2));
    }
    if out.len == 0 { return; }
    let storage = |code| -> StorageType { match code {
        0 => DType::F32, 1 => DType::F16, 2 => DType::BF16, 12 | 13 | 15 => DType::U8,
        _ => panic!("unsupported scaled_mm storage"),
    }.into() };
    let c = client(); let output_type = storage(out.dtype); let bias_type = storage(bias.dtype); let scale_type = storage(sa.dtype);
    if k == 0 {
        unsafe { zero::launch::<CudaRuntime>(&c, RudaCount::Static(u32::try_from(out.len.div_ceil(128)).unwrap(), 1, 1),
            RudaDim::new_1d(128), out.arg(), output_type); }
    } else {
        let fp8 = |code| -> StorageType { ElemType::Float(if code == 12 {FloatKind::E4M3} else {FloatKind::E5M2}).into() };
        let native = c.features().matmul.mma.contains(&MmaConfig {a_type: fp8(a.dtype), b_type: fp8(b.dtype),
            cd_type: DType::F32.into(), m: 16, n: 8, k: 32});
        let carrier = MmaConfig {a_type: DType::F16.into(), b_type: DType::F16.into(), cd_type: DType::F32.into(), m: 16, n: 16, k: 16};
        let h = &c.properties().hardware; let tile = if native {8} else {16};
        assert!(h.plane_size_min == 32 && h.plane_size_max == 32);
        assert!(native || (h.max_shared_memory_size >= 2048 && c.features().matmul.cmma.contains(&carrier)), "scaled_mm requires FP8 or FP16 tensor cores");
        assert!(m <= u32::MAX as usize - 16 && k <= u32::MAX as usize - 32 && n <= u32::MAX as usize - 16);
        assert!(n.div_ceil(tile) <= h.max_ruda_count.0 as usize && m.div_ceil(16) <= h.max_ruda_count.1 as usize);
        let grid = RudaCount::Static(n.div_ceil(tile) as u32, m.div_ceil(16) as u32, 1);
        unsafe {
            if native {
                native::launch::<CudaRuntime>(&c, grid, RudaDim::new_1d(32), a.arg(), b.arg(), sa.arg(), sb.arg(), bias.arg(), out.arg(),
                    ma as u32, mb as u32, fast, has_bias, out.dtype, fp8(a.dtype), fp8(b.dtype), output_type, bias_type, scale_type);
            } else {
                carrier::launch::<CudaRuntime>(&c, grid, RudaDim::new_1d(32), a.arg(), b.arg(), sa.arg(), sb.arg(), bias.arg(), out.arg(),
                    ma as u32, mb as u32, if a.dtype == 12 {18} else {19}, if b.dtype == 12 {18} else {19},
                    has_bias, out.dtype, output_type, bias_type, scale_type);
            }
        }
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed); finish_dispatch(&c);
}
