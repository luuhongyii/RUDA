use ruda_core::{ir::features::MmaConfig, tensor::DType};
use ruda_kernel::{dsl as kernel_dsl, dsl::{cmma, prelude::{*, barrier::{Barrier, copy_async}}}, tensor::RudaTensor};

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
        || !a.client.features().copy_async
        || !a.client.features().matmul.cmma.contains(&MmaConfig {
            a_type: DType::BF16.into(), b_type: DType::BF16.into(), cd_type: DType::F32.into(),
            m: 16, n: 16, k: 16,
        })
    { return false; }
    super::training_tile_tune::matmul(a.clone(), b.clone(), out.clone());
    true
}

#[allow(unsafe_code)]
pub(super) fn launch_tile<R: Runtime>(a: RudaTensor<R>, b: RudaTensor<R>, out: RudaTensor<R>,
    bm: usize, bn: usize, bk: usize, asynchronous: bool) -> Result<RudaTensor<R>, String> {
    let (k, n) = (a.meta.shape()[1], b.meta.shape()[1]);
    if k % bk != 0 { return Err("training K tile does not divide the reduction".into()); }
    let nt = b.meta.strides()[0] == 1;
    let lda = a.meta.strides()[0];
    let ldb = if nt { b.meta.strides()[1] } else { b.meta.strides()[0] };
    let a_span = 511 * lda + k;
    let b_span = if nt { (n - 1) * ldb + k } else { (k - 1) * ldb + n };
    let mut left = a.clone(); left.meta.shape_mut()[0] = a_span; left.meta.shape_mut()[1] = 1;
    let mut right = b.clone(); right.meta.shape_mut()[0] = b_span; right.meta.shape_mut()[1] = 1;
    unsafe { tiled::launch_unchecked::<R>(&a.client, RudaCount::Static((n / bn * (512 / bm)) as u32, 1, 1), RudaDim::new_1d((bm * 2) as u32),
        left.into_array_arg(), right.into_array_arg(), out.clone().into_array_arg(), k, n, lda, ldb, nt,
        bm, bn, bk, asynchronous,
        DType::BF16.into(), out.dtype.into()); }
    Ok(out)
}

#[ruda(launch_unchecked)]
fn tiled<F: Float, O: Float>(a: &Array<Vector<F, Const<8>>>, b: &Array<Vector<F, Const<8>>>,
    out: &mut Array<O>, #[comptime] k: usize, #[comptime] n: usize,
    #[comptime] lda: usize, #[comptime] ldb: usize, #[comptime] nt: bool,
    #[comptime] bm: usize, #[comptime] bn: usize, #[comptime] bk: usize, #[comptime] asynchronous: bool,
    #[define(F)] _input: StorageType, #[define(O)] _output: StorageType) {
    let thread = UNIT_POS as usize;
    let lane = thread % 32;
    let warp = thread / 32;
    // Adjacent CTAs reuse the weight tile; smaller tiles keep serial-K views occupied.
    let row = RUDA_POS_X as usize % (512 / bm) * bm;
    let col = RUDA_POS_X as usize / (512 / bm) * bn;
    let wm = warp / 2 * 32;
    let wn = warp % 2 * (bn / 2);
    let banks = comptime!(if asynchronous { 2usize } else { 1usize });
    let a_stage = comptime!(bm * (bk / 8 + 1));
    let b_stage = comptime!(if nt { bn * (bk / 8 + 1) } else { bk * (bn / 8 + 1) });
    let mut sa = SharedMemory::<Vector<F, Const<8>>>::new_aligned(a_stage * banks, 32usize);
    let mut sb = SharedMemory::<Vector<F, Const<8>>>::new_aligned(b_stage * banks, 32usize);
    let mut epilogue = SharedMemory::<f32>::new_aligned(bm * 16, 32usize);
    let loaded = Barrier::shared(RUDA_DIM, UNIT_POS == 0);
    let mut acc = Sequence::<cmma::Matrix<f32>>::new();
    #[unroll]
    for _ in 0usize..(bn / 16) {
        acc.push(cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,
            16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0f32));
    }
    if comptime!(asynchronous) {
        load_stage(a, b, &mut sa, &mut sb, row, col, 0usize, 0usize, lda, ldb, nt, bm, bn, bk, asynchronous);
        loaded.commit_copy_async();
    }
    for block in 0usize..k / bk {
        let bank = block % banks;
        if comptime!(asynchronous) {
            loaded.arrive_and_wait();
            if block + 1 < k / bk {
                load_stage(a, b, &mut sa, &mut sb, row, col, block + 1, 1 - bank, lda, ldb, nt, bm, bn, bk, asynchronous);
                loaded.commit_copy_async();
            }
        } else {
            load_stage(a, b, &mut sa, &mut sb, row, col, block, bank, lda, ldb, nt, bm, bn, bk, asynchronous);
            sync_ruda();
        }
        #[unroll]
        for half in 0usize..(bk / 16) {
            let mut ar = Sequence::<cmma::Matrix<Vector<F, Const<8>>>>::new();
            let mut br = Sequence::<cmma::Matrix<Vector<F, Const<8>>>>::new();
            #[unroll]
            for i in 0usize..2 {
                let offset = bank * a_stage + (wm + i * 16) * (bk / 8 + 1) + half * 2;
                ar.push(cmma::Matrix::<Vector<F, Const<8>>>::from_slice(cmma::MatrixIdent::A,
                    16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor,
                    &sa.slice(offset, a_stage * banks), (bk + 8) as u32));
            }
            #[unroll]
            for i in 0usize..(bn / 32) {
                if comptime!(nt) {
                    let offset = bank * b_stage + (wn + i * 16) * (bk / 8 + 1) + half * 2;
                    br.push(cmma::Matrix::<Vector<F, Const<8>>>::from_slice(cmma::MatrixIdent::B,
                        16usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor,
                        &sb.slice(offset, b_stage * banks), (bk + 8) as u32));
                } else {
                    let offset = bank * b_stage + half * 16 * (bn / 8 + 1) + wn / 8 + i * 2;
                    br.push(cmma::Matrix::<Vector<F, Const<8>>>::from_slice(cmma::MatrixIdent::B,
                        16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor,
                        &sb.slice(offset, b_stage * banks), (bn + 8) as u32));
                }
            }
            #[unroll]
            for i in 0usize..2 {
                #[unroll]
                for j in 0usize..(bn / 32) {
                    cmma::execute(&ar[i], &br[j], &acc[i * (bn / 32) + j], &acc[i * (bn / 32) + j]);
                }
            }
        }
        if comptime!(!asynchronous) { sync_ruda(); }
    }
    #[unroll]
    for i in 0usize..2 {
        #[unroll]
        for j in 0usize..(bn / 32) {
            cmma::store(&mut epilogue.slice_mut(warp * 256, warp * 256 + 256),
                &acc[i * (bn / 32) + j], 16u32, cmma::MatrixLayout::RowMajor);
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

#[ruda]
fn load_stage<F: Float>(a: &Array<Vector<F, Const<8>>>, b: &Array<Vector<F, Const<8>>>,
    sa: &mut SharedMemory<Vector<F, Const<8>>>, sb: &mut SharedMemory<Vector<F, Const<8>>>,
    row: usize, col: usize, block: usize, bank: usize,
    #[comptime] lda: usize, #[comptime] ldb: usize, #[comptime] nt: bool,
    #[comptime] bm: usize, #[comptime] bn: usize, #[comptime] bk: usize, #[comptime] asynchronous: bool) {
    #[unroll]
    for i in 0usize..(bk / 16) {
        let p = UNIT_POS as usize + i * (bm * 2);
        let source = ((row + p / (bk / 8)) * lda + block * bk) / 8 + p % (bk / 8);
        let target = bank * bm * (bk / 8 + 1) + (p / (bk / 8)) * (bk / 8 + 1) + p % (bk / 8);
        if comptime!(asynchronous) {
            copy_async(&a.slice(source, source + 1), &mut sa.slice_mut(target, target + 1), 8u32);
        } else { sa[target] = a[source]; }
    }
    #[unroll]
    for i in 0usize..(bn * bk / (bm * 16)) {
        let p = UNIT_POS as usize + i * (bm * 2);
        if comptime!(nt) {
            let source = ((col + p / (bk / 8)) * ldb + block * bk) / 8 + p % (bk / 8);
            let target = bank * bn * (bk / 8 + 1) + (p / (bk / 8)) * (bk / 8 + 1) + p % (bk / 8);
            if comptime!(asynchronous) {
                copy_async(&b.slice(source, source + 1), &mut sb.slice_mut(target, target + 1), 8u32);
            } else { sb[target] = b[source]; }
        } else {
            let source = ((block * bk + p / (bn / 8)) * ldb + col) / 8 + p % (bn / 8);
            let target = bank * bk * (bn / 8 + 1) + (p / (bn / 8)) * (bn / 8 + 1) + p % (bn / 8);
            if comptime!(asynchronous) {
                copy_async(&b.slice(source, source + 1), &mut sb.slice_mut(target, target + 1), 8u32);
            } else { sb[target] = b[source]; }
        }
    }
}
