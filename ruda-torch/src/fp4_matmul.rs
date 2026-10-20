use super::*;
use ruda_core::{ir::features::MmaConfig, tensor::DType};
use ruda_kernel::{dsl::prelude::Tensor, quantization::packed::decode_code};

#[ruda(launch)]
fn matmul<O: Numeric, P: Float>(
    a: &Tensor<u8>, b: &Tensor<u8>, sa: &Tensor<u8>, sb: &Tensor<u8>,
    ga: &Tensor<f32>, gb: &Tensor<f32>, bias: &Tensor<P>, out: &mut Tensor<O>,
    #[comptime] group: usize, #[comptime] global: bool, #[comptime] has_bias: bool,
    #[comptime] output_code: u32, #[define(O)] _output_type: StorageType, #[define(P)] _bias_type: StorageType,
) {
    let m = a.shape(0); let k = a.shape(1) * 2usize; let n = b.shape(1);
    let row_base = RUDA_POS_Y as usize * 16usize; let col_base = RUDA_POS_X as usize * 16usize;
    let lane = UNIT_POS as usize;
    let mut left = SharedMemory::<f16>::new_aligned(256usize, 32usize);
    let mut right = SharedMemory::<f16>::new_aligned(256usize, 32usize);
    let mut partial = SharedMemory::<f32>::new_aligned(256usize, 32usize);
    let mut total = Array::<f32>::new(8usize);
    #[unroll] for i in 0usize..8usize { total[i] = 0.0f32; }
    let acc = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize,
        cmma::MatrixLayout::Undefined, 0.0f32);
    let mut base = 0usize;
    while base < k {
        #[unroll] for i in 0usize..8usize {
            let t = lane + i * 32usize; let r = t / 16usize; let c = t % 16usize;
            let mut av = 0.0f32; let mut bv = 0.0f32;
            if row_base + r < m && base + c < k {
                let inner = base + c;
                let byte = a[(row_base + r) * a.stride(0) + inner / 2usize * a.stride(1)] as u32;
                av = decode_code((byte >> (u32::cast_from(inner % 2usize) * 4u32)) & 15u32, 14u32);
            }
            if base + r < k && col_base + c < n {
                let inner = base + r;
                let byte = b[inner / 2usize * b.stride(0) + (col_base + c) * b.stride(1)] as u32;
                bv = decode_code((byte >> (u32::cast_from(inner % 2usize) * 4u32)) & 15u32, 14u32);
            }
            left[t] = f16::cast_from(av); right[t] = f16::cast_from(bv);
        }
        sync_ruda();
        let aa = cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::A, 16usize, 16usize, 16usize,
            cmma::MatrixLayout::RowMajor, &left.to_slice(), 16u32);
        let bb = cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::B, 16usize, 16usize, 16usize,
            cmma::MatrixLayout::RowMajor, &right.to_slice(), 16u32);
        cmma::execute::<f16, f16, f32, f32, _>(&aa, &bb, &acc, &acc); sync_ruda();
        if k - base <= 16usize || (base + 16usize) % group == 0usize {
            cmma::store(&mut partial.to_slice_mut(), &acc, 16u32, cmma::MatrixLayout::RowMajor); sync_ruda();
            #[unroll] for i in 0usize..8usize {
                let t = lane + i * 32usize; let row = row_base + t / 16usize; let col = col_base + t % 16usize;
                if row < m && col < n {
                    let mut value = partial[t];
                    if comptime!(group == 32usize) {
                        value = crate::scaled_matmul::scaled_block(sa, sb, value, row, col, base, k, 6u32, 6u32);
                    } else {
                        let av = decode_code(sa[crate::scaled_matmul::block_index(row, base, k, 16usize)] as u32, 18u32);
                        let bv = decode_code(sb[crate::scaled_matmul::block_index(col, base, k, 16usize)] as u32, 18u32);
                        value = value * av * bv;
                    }
                    total[i] += value;
                }
            }
            cmma::fill(&acc, 0.0f32); sync_ruda();
        }
        if k - base <= 16usize { base = k; } else { base += 16usize; }
    }
    let mut alpha = 1.0f32;
    if comptime!(global) { if k != 0usize { alpha = ga[0] * gb[0]; } }
    #[unroll] for i in 0usize..8usize {
        let t = lane + i * 32usize; let row = row_base + t / 16usize; let col = col_base + t % 16usize;
        if row < m && col < n {
            if k == 0usize { out[row * out.stride(0) + col * out.stride(1)] = O::cast_from(0u32); }
            else { crate::scaled_matmul::store(bias, out, row, col, total[i] * alpha, has_bias, output_code); }
        }
    }
}

pub(super) fn dispatch(xs: &[View], ints: &[i64]) {
    assert_eq!(xs.len(), 8); assert_eq!(ints.len(), 3);
    let (a, b, sa, sb, ga, gb, bias, out) = (&xs[0], &xs[1], &xs[2], &xs[3], &xs[4], &xs[5], &xs[6], &xs[7]);
    let group = usize::try_from(ints[0]).unwrap(); let global = ints[1] != 0; let has_bias = ints[2] != 0;
    assert!(matches!(group, 16 | 32) && matches!(ints[1], 0 | 1) && matches!(ints[2], 0 | 1));
    assert!(group == 16 || (!global && !has_bias && out.dtype == 2));
    assert_eq!(a.dtype, 16); assert_eq!(b.dtype, 16); assert_eq!(a.shape.len(), 2); assert_eq!(b.shape.len(), 2);
    let (m, packed_k, n) = (a.shape[0], a.shape[1], b.shape[1]);
    let k = packed_k.checked_mul(2).expect("FP4 contraction size overflow");
    assert_eq!(b.shape[0], packed_k); assert_eq!(out.shape, vec![m, n]); assert!(k % 16 == 0 && n % 16 == 0);
    assert!(matches!(out.dtype, 0 | 1 | 2));
    for (s, outer) in [(sa, m), (sb, n)] {
        assert_eq!(s.dtype, if group == 16 {12} else {15});
        let count = outer.div_ceil(128).checked_mul(128)
            .and_then(|r| r.checked_mul(k.div_ceil(group * 4) * 4)).expect("FP4 scale size overflow");
        assert_eq!(s.len, count);
        let mut stride = 1usize;
        for d in (0..s.shape.len()).rev() { assert!(s.shape[d] <= 1 || s.strides[d] == stride); stride *= s.shape[d]; }
    }
    assert_eq!(ga.dtype, 0); assert_eq!(gb.dtype, 0);
    if global { assert_eq!(ga.len, 1); assert_eq!(gb.len, 1); }
    if has_bias {
        assert!(matches!(bias.dtype, 1 | 2)); assert_eq!(bias.shape, vec![n]); assert_eq!(bias.strides, vec![1]);
        assert!(out.dtype != 0 && (out.dtype != 1 || bias.dtype == 1) && (out.dtype != 2 || bias.dtype == 2));
    }
    if out.len == 0 { return; }
    let storage = |code| -> StorageType { match code {
        0 => DType::F32, 1 => DType::F16, 2 => DType::BF16,
        _ => panic!("unsupported FP4 matmul output/bias storage"),
    }.into() };
    let c = client(); let h = &c.properties().hardware;
    let mma = MmaConfig {a_type: DType::F16.into(), b_type: DType::F16.into(), cd_type: DType::F32.into(), m: 16, n: 16, k: 16};
    assert!(h.plane_size_min == 32 && h.plane_size_max == 32 && h.max_shared_memory_size >= 2048);
    assert!(c.features().matmul.cmma.contains(&mma), "FP4 matmul requires FP16 tensor cores");
    assert!(m <= u32::MAX as usize - 16 && n <= u32::MAX as usize - 16 && k <= u32::MAX as usize - 16);
    assert!(n.div_ceil(16) <= h.max_ruda_count.0 as usize && m.div_ceil(16) <= h.max_ruda_count.1 as usize);
    unsafe { matmul::launch::<CudaRuntime>(&c, RudaCount::Static(n.div_ceil(16) as u32, m.div_ceil(16) as u32, 1),
        RudaDim::new_1d(32), a.arg(), b.arg(), sa.arg(), sb.arg(), ga.arg(), gb.arg(), bias.arg(), out.arg(),
        group, global, has_bias, out.dtype, storage(out.dtype), storage(bias.dtype)); }
    LAUNCHES.fetch_add(1, Ordering::Relaxed); finish_dispatch(&c);
}
