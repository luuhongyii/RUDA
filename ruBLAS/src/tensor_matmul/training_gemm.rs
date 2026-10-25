use ruda_core::{ir::features::MmaConfig, tensor::DType};
use ruda_kernel::{dsl as kernel_dsl, dsl::{cmma, prelude::*}, tensor::RudaTensor};

/// Fixed-layout BF16 training GEMM. Accumulation stays FP32, in increasing K order.
#[allow(unsafe_code)]
pub fn matmul_bf16_training<R: Runtime>(a: &RudaTensor<R>, b: &RudaTensor<R>, out: &RudaTensor<R>) -> bool {
    if R::name(&a.client) != "cuda"
        || a.dtype != DType::BF16 || b.dtype != DType::BF16 || !matches!(out.dtype, DType::BF16 | DType::F32)
        || a.qparams.is_some() || b.qparams.is_some() || out.qparams.is_some()
        || a.meta.num_dims() != 2 || b.meta.num_dims() != 2 || out.meta.num_dims() != 2
        || !a.client.same_execution_queue(&b.client) || !a.client.same_execution_queue(&out.client)
    { return false; }
    let (m, k, n) = (a.meta.shape()[0], a.meta.shape()[1], b.meta.shape()[1]);
    let nt = b.meta.strides()[0] == 1;
    let lda = a.meta.strides()[0];
    let ldb = if nt { b.meta.strides()[1] } else { b.meta.strides()[0] };
    if m != 512 || k == 0 || k % 32 != 0 || n < 128 || n % 128 != 0
        || b.meta.shape()[0] != k || a.meta.strides()[1] != 1 || lda < k || lda % 8 != 0
        || (!nt && b.meta.strides()[1] != 1) || ldb < (if nt { k } else { n }) || ldb % 8 != 0
        || out.meta.shape()[..] != [m, n] || out.meta.strides()[..] != [n, 1]
        || a.handle.offset_start.unwrap_or(0) % 16 != 0 || b.handle.offset_start.unwrap_or(0) % 16 != 0
    { return false; }
    let Some(a_span) = (m - 1).checked_mul(lda).and_then(|v| v.checked_add(k)) else { return false; };
    let Some(b_span) = (if nt { n - 1 } else { k - 1 }).checked_mul(ldb)
        .and_then(|v| v.checked_add(if nt { k } else { n })) else { return false; };
    if a_span > u32::MAX as usize || b_span > u32::MAX as usize
        || m.checked_mul(n).is_none_or(|v| v > u32::MAX as usize)
        || !a.client.features().matmul.cmma.contains(&MmaConfig {
            a_type: DType::BF16.into(), b_type: DType::BF16.into(), cd_type: DType::F32.into(),
            m: 16, n: 16, k: 16,
        })
    { return false; }
    // Array bindings must describe the physical span, including serial-K view gaps.
    let mut left = a.clone(); left.meta.shape_mut()[0] = a_span; left.meta.shape_mut()[1] = 1;
    let mut right = b.clone(); right.meta.shape_mut()[0] = b_span; right.meta.shape_mut()[1] = 1;
    // Full tiles, aligned bindings and checked physical spans bound every access.
    unsafe { tiled::launch_unchecked::<R>(&a.client, RudaCount::Static((n / 128 * 4) as u32, 1, 1), RudaDim::new_1d(256),
        left.into_array_arg(), right.into_array_arg(), out.clone().into_array_arg(), k, n, lda, ldb, nt,
        DType::BF16.into(), out.dtype.into()); }
    true
}

#[ruda(launch_unchecked)]
fn tiled<F: Float, O: Float>(a: &Array<Vector<F, Const<8>>>, b: &Array<Vector<F, Const<8>>>,
    out: &mut Array<O>, #[comptime] k: usize, #[comptime] n: usize,
    #[comptime] lda: usize, #[comptime] ldb: usize, #[comptime] nt: bool,
    #[define(F)] _input: StorageType, #[define(O)] _output: StorageType) {
    let thread = UNIT_POS as usize;
    let lane = thread % 32;
    let warp = thread / 32;
    // Adjacent CTAs reuse the same weight tile across all four M blocks.
    let row = RUDA_POS_X as usize % 4 * 128;
    let col = RUDA_POS_X as usize / 4 * 128;
    let wm = warp / 2 * 32;
    let wn = warp % 2 * 64;
    let mut sa = SharedMemory::<Vector<F, Const<8>>>::new_aligned(640usize, 32usize);
    let mut sb = SharedMemory::<Vector<F, Const<8>>>::new_aligned(if nt { 640usize } else { 544usize }, 32usize);
    let mut epilogue = SharedMemory::<f32>::new_aligned(2048usize, 32usize);
    let mut acc = Sequence::<cmma::Matrix<f32>>::new();
    #[unroll]
    for _ in 0usize..8 {
        acc.push(cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,
            16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0f32));
    }
    for block in 0usize..k / 32 {
        #[unroll]
        for i in 0usize..2 {
            let p = thread + i * 256;
            sa[(p / 4) * 5 + p % 4] = a[((row + p / 4) * lda + block * 32) / 8 + p % 4];
            if comptime!(nt) {
                sb[(p / 4) * 5 + p % 4] = b[((col + p / 4) * ldb + block * 32) / 8 + p % 4];
            } else {
                sb[(p / 16) * 17 + p % 16] = b[((block * 32 + p / 16) * ldb + col) / 8 + p % 16];
            }
        }
        sync_ruda();
        #[unroll]
        for half in 0usize..2 {
            let mut ar = Sequence::<cmma::Matrix<Vector<F, Const<8>>>>::new();
            let mut br = Sequence::<cmma::Matrix<Vector<F, Const<8>>>>::new();
            #[unroll]
            for i in 0usize..2 {
                let offset = (wm + i * 16) * 5 + half * 2;
                ar.push(cmma::Matrix::<Vector<F, Const<8>>>::from_slice(cmma::MatrixIdent::A,
                    16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor,
                    &sa.slice(offset, 640usize), 40u32));
            }
            #[unroll]
            for i in 0usize..4 {
                if comptime!(nt) {
                    let offset = (wn + i * 16) * 5 + half * 2;
                    br.push(cmma::Matrix::<Vector<F, Const<8>>>::from_slice(cmma::MatrixIdent::B,
                        16usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor,
                        &sb.slice(offset, 640usize), 40u32));
                } else {
                    let offset = half * 16 * 17 + wn / 8 + i * 2;
                    br.push(cmma::Matrix::<Vector<F, Const<8>>>::from_slice(cmma::MatrixIdent::B,
                        16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor,
                        &sb.slice(offset, 544usize), 136u32));
                }
            }
            #[unroll]
            for i in 0usize..2 {
                #[unroll]
                for j in 0usize..4 {
                    cmma::execute(&ar[i], &br[j], &acc[i * 4 + j], &acc[i * 4 + j]);
                }
            }
        }
        sync_ruda();
    }
    #[unroll]
    for i in 0usize..2 {
        #[unroll]
        for j in 0usize..4 {
            cmma::store(&mut epilogue.slice_mut(warp * 256, warp * 256 + 256),
                &acc[i * 4 + j], 16u32, cmma::MatrixLayout::RowMajor);
            sync_plane();
            #[unroll]
            for t in 0usize..8 {
                let pos = lane + t * 32;
                out[(row + wm + i * 16 + pos / 16) * n + col + wn + j * 16 + pos % 16]
                    = O::cast_from(epilogue[warp * 256 + pos]);
            }
            sync_plane();
        }
    }
}
