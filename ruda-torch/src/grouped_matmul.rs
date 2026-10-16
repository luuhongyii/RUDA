use super::*;
use ruda_core::{ir::{ElemType, FloatKind, features::MmaConfig}, tensor::DType};
use ruda_kernel::{dsl::prelude::Tensor, quantization::packed::decode_code};

#[ruda(launch)]
fn prepare(a: &Tensor<u8>, b: &Tensor<u8>, offsets: &Tensor<i32>, sa: &Tensor<u8>, sb: &Tensor<u8>,
    out: &Tensor<u8>, metadata: &mut Tensor<u32>, status: &mut Tensor<u32>, groups: usize,
    #[comptime] mode: u32, #[comptime] kind: u32, #[comptime] tile_n: usize,
) {
    let ad = comptime!(if mode == 0 || mode == 2 {1usize} else {0usize});
    let bd = comptime!(if mode == 0 || mode == 1 {1usize} else {0usize});
    let od = comptime!(if mode == 0 || mode == 3 {1usize} else {0usize});
    let pack = comptime!(if kind >= 3 {2usize} else {1usize});
    let scale_block = comptime!(if kind == 4 {16usize} else {32usize});
    let full_m = a.shape(ad); let full_n = b.shape(bd + 1usize);
    let full_k = a.shape(ad + 1usize) * pack; let other_k = b.shape(bd) * pack;
    let mut previous = 0usize; let mut tiles = 0u64; let mut sa_base = 0u64; let mut sb_base = 0u64;
    status[0] = 0u32; metadata[groups * 10usize] = 0u32;
    for g in 0..groups {
        let mut m = full_m; let mut n = full_n; let mut k = full_k;
        let mut ao = 0usize; let mut bo = 0usize; let mut co = 0usize;
        let mut end = previous;
        if comptime!(mode != 0) {
            let signed = offsets[g];
            if signed < 0i32 { status[0] = 1u32; terminate!(); }
            end = usize::cast_from(signed);
            let mut limit = usize::min(full_k, other_k);
            if comptime!(mode == 1) { limit = full_m; }
            if comptime!(mode == 2) { limit = full_n; }
            if end < previous || end > limit { status[0] = 1u32; terminate!(); }
        }
        if comptime!(mode == 0) {
            ao = g * a.stride(0); bo = g * b.stride(0); co = g * out.stride(0);
        } else if comptime!(mode == 1) {
            m = end - previous; ao = previous * a.stride(ad); bo = g * b.stride(0); co = previous * out.stride(od);
        } else if comptime!(mode == 2) {
            n = end - previous; ao = g * a.stride(0); bo = previous * b.stride(bd + 1usize); co = previous * out.stride(od + 1usize);
        } else {
            k = end - previous; ao = previous / pack * a.stride(ad + 1usize); bo = previous / pack * b.stride(bd); co = g * out.stride(0);
        }
        let mut spa = 0u64; let mut spb = 0u64;
        if comptime!(kind == 1) {
            if comptime!(mode == 0) { spa = u64::cast_from(g * sa.stride(0)); spb = u64::cast_from(g * sb.stride(0)); }
            else if comptime!(mode == 1) { spa = u64::cast_from(previous); spb = u64::cast_from(g * sb.stride(0)); }
            else if comptime!(mode == 2) { spa = u64::cast_from(g * sa.stride(0)); spb = u64::cast_from(previous); }
            else { spa = u64::cast_from(g) * u64::cast_from(m); spb = u64::cast_from(g) * u64::cast_from(n); }
        }
        if comptime!(kind >= 2) {
            if comptime!(mode == 3) { if k % scale_block != 0usize { status[0] = 2u32; terminate!(); } }
            let width = u64::cast_from(k / (scale_block * 4usize) + usize::cast_from(k % (scale_block * 4usize) != 0usize)) * 4u64;
            let height_a = (u64::cast_from(m) + 127u64) / 128u64 * 128u64;
            let height_b = (u64::cast_from(n) + 127u64) / 128u64 * 128u64;
            spa = sa_base; spb = sb_base;
            sa_base += height_a * width; sb_base += height_b * width;
            if sa_base > u64::cast_from(sa.len()) || sb_base > u64::cast_from(sb.len()) { status[0] = 3u32; terminate!(); }
        }
        let nt = n / tile_n + usize::cast_from(n % tile_n != 0usize);
        let mt = m / 16usize + usize::cast_from(m % 16usize != 0usize);
        let p = g * 10usize;
        metadata[p] = u32::cast_from(tiles); metadata[p + 1usize] = u32::cast_from(m);
        metadata[p + 2usize] = u32::cast_from(n); metadata[p + 3usize] = u32::cast_from(k);
        metadata[p + 4usize] = u32::cast_from(ao); metadata[p + 5usize] = u32::cast_from(bo);
        metadata[p + 6usize] = u32::cast_from(co); metadata[p + 7usize] = u32::cast_from(spa);
        metadata[p + 8usize] = u32::cast_from(spb); metadata[p + 9usize] = u32::cast_from(nt);
        tiles += u64::cast_from(mt) * u64::cast_from(nt);
        if tiles > 4294967295u64 { status[0] = 4u32; terminate!(); }
        previous = end;
    }
    metadata[groups * 10usize] = u32::cast_from(tiles);
}

#[ruda]
fn group_for(metadata: &Tensor<u32>, block: usize, groups: usize) -> usize {
    let mut lo = 0usize; let mut hi = groups;
    while lo < hi {
        let mid = lo + (hi - lo) / 2usize;
        if usize::cast_from(metadata[(mid + 1usize) * 10usize]) <= block { lo = mid + 1usize; }
        else { hi = mid; }
    }
    lo
}

#[ruda]
fn load<I: Numeric>(x: &Tensor<I>, base: usize, row: usize, column: usize,
    #[comptime] axis: usize, #[comptime] code: u32, #[comptime] right: bool,
) -> f32 {
    if comptime!(code == 16) {
        let inner = if comptime!(right) {row} else {column};
        let outer = if comptime!(right) {column} else {row};
        let inner_stride = if comptime!(right) {x.stride(axis)} else {x.stride(axis + 1usize)};
        let outer_stride = if comptime!(right) {x.stride(axis + 1usize)} else {x.stride(axis)};
        let byte = u32::cast_from(x[base + outer * outer_stride + inner / 2usize * inner_stride]);
        decode_code((byte >> (u32::cast_from(inner % 2usize) * 4u32)) & 15u32, 14u32)
    } else {
        let value = x[base + row * x.stride(axis) + column * x.stride(axis + 1usize)];
        if comptime!(code == 12) { decode_code(u32::cast_from(value), 18u32) }
        else { f32::cast_from(value) }
    }
}

#[ruda]
fn block_scale<S: Numeric>(sa: &Tensor<S>, sb: &Tensor<S>, metadata: &Tensor<u32>, p: usize,
    value: f32, row: usize, column: usize, inner: usize, k: usize, #[comptime] kind: u32,
) -> f32 {
    let size = comptime!(if kind == 4 {16usize} else {32usize});
    let ia = usize::cast_from(metadata[p + 7usize]) + crate::scaled_matmul::block_index(row, inner, k, size);
    let ib = usize::cast_from(metadata[p + 8usize]) + crate::scaled_matmul::block_index(column, inner, k, size);
    if comptime!(kind == 4) {
        value * decode_code(u32::cast_from(sa[ia]), 18u32) * decode_code(u32::cast_from(sb[ib]), 18u32)
    } else {
        let ea = u32::cast_from(sa[ia]); let eb = u32::cast_from(sb[ib]);
        let mut result = f32::reinterpret(0x7fc00000u32);
        if ea != 255u32 && eb != 255u32 {
            let power = f64::reinterpret(u64::cast_from(ea + eb + 769u32) << 52u32);
            result = f32::cast_from(f64::cast_from(value) * power);
        }
        result
    }
}

#[ruda]
fn store<S: Numeric, O: Float>(sa: &Tensor<S>, sb: &Tensor<S>, alpha: &Tensor<f32>, metadata: &Tensor<u32>,
    out: &mut Tensor<O>, group: usize, row: usize, column: usize, value: f32,
    #[comptime] kind: u32, #[comptime] mode: u32,
) {
    let p = group * 10usize;
    let mut value = value;
    if comptime!(kind == 1) {
        value = value * f32::cast_from(sa[usize::cast_from(metadata[p + 7usize]) + row])
            * f32::cast_from(sb[usize::cast_from(metadata[p + 8usize]) + column]);
    }
    if comptime!(kind == 4) { value *= alpha[group]; }
    let od = comptime!(if mode == 0 || mode == 3 {1usize} else {0usize});
    let base = usize::cast_from(metadata[p + 6usize]);
    out[base + row * out.stride(od) + column * out.stride(od + 1usize)] = O::cast_from(value);
}

#[ruda(launch)]
fn simt<I: Numeric, S: Numeric, O: Float>(
    a: &Tensor<I>, b: &Tensor<I>, sa: &Tensor<S>, sb: &Tensor<S>, alpha: &Tensor<f32>,
    metadata: &Tensor<u32>, out: &mut Tensor<O>, groups: usize,
    #[comptime] mode: u32, #[comptime] kind: u32, #[comptime] code: u32,
    #[define(I)] _input_type: StorageType, #[define(S)] _scale_type: StorageType, #[define(O)] _output_type: StorageType,
) {
    let block = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    if block >= usize::cast_from(metadata[groups * 10usize]) { terminate!(); }
    let group = group_for(metadata, block, groups); let p = group * 10usize;
    let tile = block - usize::cast_from(metadata[p]); let nt = usize::cast_from(metadata[p + 9usize]);
    let tid = UNIT_POS as usize; let ty = tid / 16usize; let tx = tid % 16usize;
    let row_base = tile / nt * 16usize; let col_base = tile % nt * 16usize;
    let row = row_base + ty; let col = col_base + tx;
    let m = usize::cast_from(metadata[p + 1usize]); let n = usize::cast_from(metadata[p + 2usize]); let k = usize::cast_from(metadata[p + 3usize]);
    let ao = usize::cast_from(metadata[p + 4usize]); let bo = usize::cast_from(metadata[p + 5usize]);
    let ad = comptime!(if mode == 0 || mode == 2 {1usize} else {0usize});
    let bd = comptime!(if mode == 0 || mode == 1 {1usize} else {0usize});
    let mut left = SharedMemory::<f32>::new(256usize); let mut right = SharedMemory::<f32>::new(256usize);
    let mut sum = 0.0f32; let mut total = 0.0f32; let mut base = 0usize;
    while base < k {
        let mut av = 0.0f32; let mut bv = 0.0f32;
        if row < m && base + tx < k { av = load::<I>(a, ao, row, base + tx, ad, code, false); }
        if base + ty < k && col < n { bv = load::<I>(b, bo, base + ty, col, bd, code, true); }
        left[tid] = av; right[tid] = bv; sync_ruda();
        for inner in 0usize..16usize { sum = fma(left[ty * 16usize + inner], right[inner * 16usize + tx], sum); }
        sync_ruda();
        if comptime!(kind >= 2) {
            let size = comptime!(if kind == 4 {16usize} else {32usize});
            if k - base <= 16usize || (base + 16usize) % size == 0usize {
                if row < m && col < n { total += block_scale::<S>(sa, sb, metadata, p, sum, row, col, base, k, kind); }
                sum = 0.0f32;
            }
        }
        if k - base <= 16usize { base = k; } else { base += 16usize; }
    }
    if comptime!(kind < 2) { total = sum; }
    if row < m && col < n { store::<S, O>(sa, sb, alpha, metadata, out, group, row, col, total, kind, mode); }
}

#[ruda(launch)]
fn carrier<I: Numeric, T: Float, S: Numeric, O: Float>(
    a: &Tensor<I>, b: &Tensor<I>, sa: &Tensor<S>, sb: &Tensor<S>, alpha: &Tensor<f32>,
    metadata: &Tensor<u32>, out: &mut Tensor<O>, groups: usize,
    #[comptime] mode: u32, #[comptime] kind: u32, #[comptime] code: u32,
    #[define(I)] _input_type: StorageType, #[define(T)] _carrier_type: StorageType,
    #[define(S)] _scale_type: StorageType, #[define(O)] _output_type: StorageType,
) {
    let block = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    if block >= usize::cast_from(metadata[groups * 10usize]) { terminate!(); }
    let group = group_for(metadata, block, groups); let p = group * 10usize;
    let tile = block - usize::cast_from(metadata[p]); let nt = usize::cast_from(metadata[p + 9usize]);
    let row_base = tile / nt * 16usize; let col_base = tile % nt * 16usize; let lane = UNIT_POS as usize;
    let m = usize::cast_from(metadata[p + 1usize]); let n = usize::cast_from(metadata[p + 2usize]); let k = usize::cast_from(metadata[p + 3usize]);
    let ao = usize::cast_from(metadata[p + 4usize]); let bo = usize::cast_from(metadata[p + 5usize]);
    let ad = comptime!(if mode == 0 || mode == 2 {1usize} else {0usize});
    let bd = comptime!(if mode == 0 || mode == 1 {1usize} else {0usize});
    let mut left = SharedMemory::<T>::new_aligned(256usize, 32usize);
    let mut right = SharedMemory::<T>::new_aligned(256usize, 32usize);
    let mut partial = SharedMemory::<f32>::new_aligned(256usize, 32usize);
    let mut total = Array::<f32>::new(8usize);
    #[unroll] for i in 0usize..8usize { total[i] = 0.0f32; }
    let acc = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0f32);
    let mut base = 0usize;
    while base < k {
        #[unroll] for i in 0usize..8usize {
            let t = lane + i * 32usize; let r = t / 16usize; let c = t % 16usize;
            let mut av = 0.0f32; let mut bv = 0.0f32;
            if row_base + r < m && base + c < k { av = load::<I>(a, ao, row_base + r, base + c, ad, code, false); }
            if base + r < k && col_base + c < n { bv = load::<I>(b, bo, base + r, col_base + c, bd, code, true); }
            left[t] = T::cast_from(av); right[t] = T::cast_from(bv);
        }
        sync_ruda();
        let aa = cmma::Matrix::<T>::from_slice(cmma::MatrixIdent::A, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor, &left.to_slice(), 16u32);
        let bb = cmma::Matrix::<T>::from_slice(cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor, &right.to_slice(), 16u32);
        cmma::execute::<T, T, f32, f32>(&aa, &bb, &acc, &acc); sync_ruda();
        let mut end_group = k - base <= 16usize;
        if comptime!(kind >= 2) {
            let size = comptime!(if kind == 4 {16usize} else {32usize});
            end_group = end_group || (base + 16usize) % size == 0usize;
        }
        if end_group {
            cmma::store(&mut partial.to_slice_mut(), &acc, 16u32, cmma::MatrixLayout::RowMajor); sync_ruda();
            #[unroll] for i in 0usize..8usize {
                let t = lane + i * 32usize; let row = row_base + t / 16usize; let col = col_base + t % 16usize;
                if row < m && col < n {
                    if comptime!(kind >= 2) { total[i] += block_scale::<S>(sa, sb, metadata, p, partial[t], row, col, base, k, kind); }
                    else { total[i] = partial[t]; }
                }
            }
            cmma::fill(&acc, 0.0f32); sync_ruda();
        }
        if k - base <= 16usize { base = k; } else { base += 16usize; }
    }
    #[unroll] for i in 0usize..8usize {
        let t = lane + i * 32usize; let row = row_base + t / 16usize; let col = col_base + t % 16usize;
        if row < m && col < n { store::<S, O>(sa, sb, alpha, metadata, out, group, row, col, total[i], kind, mode); }
    }
}

#[ruda(launch)]
fn fp8<A: Float, S: Numeric>(a: &Tensor<u8>, b: &Tensor<u8>, sa: &Tensor<S>, sb: &Tensor<S>, alpha: &Tensor<f32>,
    metadata: &Tensor<u32>, out: &mut Tensor<bf16>, groups: usize,
    #[comptime] mode: u32, #[comptime] kind: u32, #[comptime] fast: bool,
    #[define(A)] _input_type: StorageType, #[define(S)] _scale_type: StorageType,
) {
    let block = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    if block >= usize::cast_from(metadata[groups * 10usize]) { terminate!(); }
    let group_index = group_for(metadata, block, groups); let p = group_index * 10usize;
    let tile = block - usize::cast_from(metadata[p]); let nt = usize::cast_from(metadata[p + 9usize]);
    let row_base = tile / nt * 16usize; let col_base = tile % nt * 8usize; let lane = UNIT_POS_PLANE;
    let m = usize::cast_from(metadata[p + 1usize]); let n = usize::cast_from(metadata[p + 2usize]); let k = usize::cast_from(metadata[p + 3usize]);
    let ao = usize::cast_from(metadata[p + 4usize]); let bo = usize::cast_from(metadata[p + 5usize]);
    let ad = comptime!(if mode == 0 || mode == 2 {1usize} else {0usize});
    let bd = comptime!(if mode == 0 || mode == 1 {1usize} else {0usize});
    let def = cmma::MmaDefinition::<A, A, f32>::new(16usize, 8usize, 32usize);
    let va = def.vector_size(cmma::MatrixIdent::A); let vb = def.vector_size(cmma::MatrixIdent::B); let vc = def.vector_size(cmma::MatrixIdent::Accumulator);
    let size!(NA) = va; let size!(NB) = vb; let size!(NC) = vc;
    let ca = def.vectors_per_lane(cmma::MatrixIdent::A); let cb = def.vectors_per_lane(cmma::MatrixIdent::B); let cc = def.vectors_per_lane(cmma::MatrixIdent::Accumulator);
    let mut ra = Array::<Vector<A, NA>>::new(ca); let mut rb = Array::<Vector<A, NB>>::new(cb);
    let mut rc = Array::<Vector<f32, NC>>::new(cc); let mut total = Array::<Vector<f32, NC>>::new(cc);
    #[unroll] for i in 0..cc { rc[i] = Vector::cast_from(0.0f32); total[i] = Vector::cast_from(0.0f32); }
    let mut base = 0usize;
    while base < k {
        #[unroll] for i in 0..ca {
            let mut reg = Vector::empty();
            #[unroll] for j in 0..va {
                let (r, c) = def.position_of_nth(lane, (i * va + j) as u32, cmma::MatrixIdent::A);
                let row = row_base + r as usize; let col = base + c as usize; let mut code = 0u8;
                if row < m && col < k { code = a[ao + row * a.stride(ad) + col * a.stride(ad + 1usize)]; }
                reg[j] = A::reinterpret(code);
            }
            ra[i] = reg;
        }
        #[unroll] for i in 0..cb {
            let mut reg = Vector::empty();
            #[unroll] for j in 0..vb {
                let (r, c) = def.position_of_nth(lane, (i * vb + j) as u32, cmma::MatrixIdent::B);
                let row = base + r as usize; let col = col_base + c as usize; let mut code = 0u8;
                if row < k && col < n { code = b[bo + row * b.stride(bd) + col * b.stride(bd + 1usize)]; }
                reg[j] = A::reinterpret(code);
            }
            rb[i] = reg;
        }
        def.execute_inplace(&ra, &rb, &mut rc);
        if comptime!(kind == 2) {
            #[unroll] for i in 0..cc {
                let value = rc[i]; let mut sum = total[i];
                #[unroll] for j in 0..vc {
                    let (r, c) = def.position_of_nth(lane, (i * vc + j) as u32, cmma::MatrixIdent::Accumulator);
                    let row = row_base + r as usize; let col = col_base + c as usize;
                    if row < m && col < n { sum[j] += block_scale::<S>(sa, sb, metadata, p, value[j], row, col, base, k, kind); }
                }
                total[i] = sum; rc[i] = Vector::cast_from(0.0f32);
            }
        } else if comptime!(!fast) {
            #[unroll] for i in 0..cc { total[i] = total[i] + rc[i]; rc[i] = Vector::cast_from(0.0f32); }
        }
        if k - base <= 32usize { base = k; } else { base += 32usize; }
    }
    #[unroll] for i in 0..cc {
        let mut reg = total[i]; if comptime!(kind == 1 && fast) { reg = rc[i]; }
        #[unroll] for j in 0..vc {
            let (r, c) = def.position_of_nth(lane, (i * vc + j) as u32, cmma::MatrixIdent::Accumulator);
            let row = row_base + r as usize; let col = col_base + c as usize;
            if row < m && col < n { store::<S, A>(sa, sb, alpha, metadata, out, group_index, row, col, reg[j], kind, mode); }
        }
    }
}

pub(super) fn dispatch(xs: &[View], ints: &[i64]) {
    assert_eq!(xs.len(), 7); assert_eq!(ints.len(), 3);
    let (a, b, offsets, sa, sb, alpha, out) = (&xs[0], &xs[1], &xs[2], &xs[3], &xs[4], &xs[5], &xs[6]);
    let (mode, kind, fast) = (ints[0], ints[1], ints[2] != 0);
    assert!((0..=3).contains(&mode) && (0..=4).contains(&kind) && matches!(ints[2], 0 | 1));
    let ad = if matches!(mode, 0 | 2) {1} else {0}; let bd = if matches!(mode, 0 | 1) {1} else {0};
    assert_eq!(a.shape.len(), ad + 2); assert_eq!(b.shape.len(), bd + 2); assert_eq!(b.dtype, a.dtype);
    assert_eq!(offsets.dtype, 5); assert_eq!(offsets.shape.len(), 1); assert!(offsets.len <= 1 || offsets.strides[0] == 1);
    let groups = if mode == 0 { assert_eq!(offsets.len, 0); a.shape[0] } else {offsets.len};
    if ad == 1 { assert_eq!(a.shape[0], groups); } if bd == 1 { assert_eq!(b.shape[0], groups); }
    let pack = if kind >= 3 {2usize} else {1usize};
    let (m, n, k) = (a.shape[ad], b.shape[bd + 1], a.shape[ad + 1].checked_mul(pack).unwrap());
    if mode != 3 { assert_eq!(a.shape[ad + 1], b.shape[bd]); }
    assert_eq!(out.shape, if matches!(mode, 0 | 3) {vec![groups, m, n]} else {vec![m, n]});
    let padded = n.div_ceil(16 / element_bytes(out.dtype)) * (16 / element_bytes(out.dtype));
    assert_eq!(out.strides, if matches!(mode, 0 | 3) {vec![m * padded, padded, 1]} else {vec![padded, 1]});
    assert_eq!(alpha.dtype, 0);
    if kind == 0 {
        assert!(matches!(a.dtype, 0 | 1 | 2)); assert_eq!(out.dtype, a.dtype); assert_eq!(sa.dtype, 0); assert_eq!(sb.dtype, 0);
    } else {
        assert_eq!(a.dtype, if kind >= 3 {16} else {12}); assert_eq!(out.dtype, 2);
        assert_eq!(sa.dtype, if kind == 1 {0} else if kind == 4 {12} else {15}); assert_eq!(sb.dtype, sa.dtype);
        assert!(a.shape[ad + 1] % 16 == 0 && b.shape[bd] % 16 == 0 && n % 16 == 0);
        if kind == 1 {
            for (s, axis, outer) in [(sa, ad, m), (sb, bd, n)] {
                if axis == 1 { assert_eq!(s.shape, vec![groups, outer]); assert_eq!(s.strides[1], 1); }
                else { assert_eq!(s.shape, vec![outer.checked_mul(if mode == 3 {groups} else {1}).unwrap()]); assert!(s.len <= 1 || s.strides[0] == 1); }
            }
        } else {
            assert!(matches!(mode, 1 | 3));
            for s in [sa, sb] {
                assert_eq!(s.shape.len(), 2);
                let mut stride = 1usize;
                for d in (0..s.shape.len()).rev() { assert!(s.shape[d] <= 1 || s.strides[d] == stride); stride *= s.shape[d]; }
            }
            if kind == 4 { assert!(alpha.len >= groups); assert_eq!(alpha.shape.len(), 1); assert!(groups <= 1 || alpha.strides[0] == 1); }
        }
    }
    if groups == 0 || out.len == 0 { return; }
    assert!(m <= u32::MAX as usize - 128 && n <= u32::MAX as usize - 128 && k <= u32::MAX as usize - 128);
    assert!(b.shape[bd].checked_mul(pack).unwrap() <= u32::MAX as usize - 128);
    let words = groups.checked_add(1).and_then(|g| g.checked_mul(10)).expect("grouped matmul metadata size overflow");
    assert!(words <= u32::MAX as usize);
    let c = client(); let h = &c.properties().hardware;
    let fp8_type: StorageType = ElemType::Float(FloatKind::E4M3).into();
    let native = matches!(kind, 1 | 2) && c.features().matmul.mma.contains(&MmaConfig {a_type: fp8_type, b_type: fp8_type,
        cd_type: DType::F32.into(), m: 16, n: 8, k: 32});
    let carrier_type: StorageType = if a.dtype == 2 {DType::BF16.into()} else {DType::F16.into()};
    let tensor_core = a.dtype != 0 && c.features().matmul.cmma.contains(&MmaConfig {a_type: carrier_type, b_type: carrier_type,
        cd_type: DType::F32.into(), m: 16, n: 16, k: 16});
    assert!(h.plane_size_min == 32 && h.plane_size_max == 32);
    assert!(native || h.max_shared_memory_size >= 2048);
    let tile_n = if native {8usize} else {16usize};
    let blocks = match mode {
        1 => m.div_ceil(16).checked_add(groups).and_then(|v| v.checked_mul(n.div_ceil(tile_n))),
        2 => n.div_ceil(tile_n).checked_add(groups).and_then(|v| v.checked_mul(m.div_ceil(16))),
        _ => groups.checked_mul(m.div_ceil(16)).and_then(|v| v.checked_mul(n.div_ceil(tile_n))),
    }.expect("grouped matmul tile count overflow");
    assert!(blocks > 0 && blocks <= u32::MAX as usize);
    let x = blocks.min(131072).min(h.max_ruda_count.0 as usize); assert!(x > 0);
    assert!(blocks.div_ceil(x) <= h.max_ruda_count.1 as usize);
    let metadata = View::packed(c.empty(words.checked_mul(4).unwrap()), vec![words], 5);
    let status = View::packed(c.empty(4), vec![1], 5);
    unsafe { prepare::launch::<CudaRuntime>(&c, RudaCount::Static(1, 1, 1), RudaDim::new_1d(1),
        a.arg(), b.arg(), offsets.arg(), sa.arg(), sb.arg(), out.arg(), metadata.arg(), status.arg(), groups, mode as u32, kind as u32, tile_n); }
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    if mode != 0 {
        sync(&c); let bytes = c.read_one(status.handle.clone()).expect("grouped matmul status readback failed");
        DOWNLOAD.fetch_add(4, Ordering::Relaxed); assert_eq!(bytes.len(), 4);
        match u32::from_le_bytes(bytes[..4].try_into().unwrap()) {
            0 => (), 1 => panic!("grouped matmul offsets must be nonnegative, nondecreasing and within the grouped dimension"),
            2 => panic!("block-scaled grouped matmul K groups must be divisible by the scale block size"),
            3 => panic!("grouped matmul scale storage is too small for the padded groups"),
            _ => panic!("grouped matmul tile count exceeds the addressable range"),
        }
    }
    let storage = |code| -> StorageType { match code {
        0 => DType::F32, 1 => DType::F16, 2 => DType::BF16, 12 | 15 | 16 => DType::U8,
        _ => panic!("unsupported grouped matmul storage"),
    }.into() };
    let grid = RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1);
    unsafe {
        if native {
            fp8::launch::<CudaRuntime>(&c, grid, RudaDim::new_1d(32), a.arg(), b.arg(), sa.arg(), sb.arg(), alpha.arg(), metadata.arg(), out.arg(),
                groups, mode as u32, kind as u32, fast, fp8_type, storage(sa.dtype));
        } else if tensor_core {
            carrier::launch::<CudaRuntime>(&c, grid, RudaDim::new_1d(32), a.arg(), b.arg(), sa.arg(), sb.arg(), alpha.arg(), metadata.arg(), out.arg(),
                groups, mode as u32, kind as u32, a.dtype, storage(a.dtype), carrier_type, storage(sa.dtype), storage(out.dtype));
        } else {
            simt::launch::<CudaRuntime>(&c, grid, RudaDim::new_1d(256), a.arg(), b.arg(), sa.arg(), sb.arg(), alpha.arg(), metadata.arg(), out.arg(),
                groups, mode as u32, kind as u32, a.dtype, storage(a.dtype), storage(sa.dtype), storage(out.dtype));
        }
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed); finish_dispatch(&c);
}
