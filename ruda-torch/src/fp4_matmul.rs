use super::*;
use ruda_core::{ir::features::MmaConfig, tensor::DType};
use ruda_kernel::{dsl::prelude::Tensor, quantization::packed::decode_code};

#[ruda(launch)]
fn matmul<O: Numeric, P: Float>(
    a: &Tensor<u8>, b: &Tensor<u8>, sa: &Tensor<u8>, sb: &Tensor<u8>,
    ga: &Tensor<f32>, gb: &Tensor<f32>, bias: &Tensor<P>, out: &mut Tensor<O>,
    #[comptime] group: usize, #[comptime] global: bool, #[comptime] has_bias: bool,
    #[comptime] split_groups: usize, #[comptime] split: bool,
    #[comptime] tile_rows: usize, #[comptime] tile_cols: usize,
    #[comptime] output_code: u32, #[define(O)] _output_type: StorageType, #[define(P)] _bias_type: StorageType,
) {
    let m = a.shape(0); let k = a.shape(1) * 2usize; let n = b.shape(1);
    let tiles_m = m / (16usize * tile_rows) + usize::cast_from(m % (16usize * tile_rows) != 0usize);
    let tiles_n = n / (16usize * tile_cols) + usize::cast_from(n % (16usize * tile_cols) != 0usize);
    let tile = RUDA_POS_Y as usize * tiles_n + RUDA_POS_X as usize;
    let position = crate::matmul::tile_position(tile,tiles_m,tiles_n);
    let block_row = position.0 * (16usize * tile_rows);
    let block_col = position.1 * (16usize * tile_cols);
    let tid = UNIT_POS as usize; let lane = tid % 32usize; let warp = tid / 32usize;
    let warp_row = warp / tile_cols; let warp_col = warp % tile_cols;
    let row_base = block_row + warp_row * 16usize; let col_base = block_col + warp_col * 16usize;
    let threads = comptime!(32 * tile_rows * tile_cols);
    let mut left = SharedMemory::<f16>::new_aligned(256usize * tile_rows, 32usize);
    let mut right = SharedMemory::<f16>::new_aligned(256usize * tile_cols, 32usize);
    let mut partial = SharedMemory::<f32>::new_aligned(256usize * tile_rows * tile_cols, 32usize);
    let mut total = Array::<f32>::new(8usize);
    #[unroll] for i in 0usize..8usize { total[i] = 0.0f32; }
    let acc = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize,
        cmma::MatrixLayout::Undefined, 0.0f32);
    let split_start = RUDA_POS_Z as usize * split_groups * group;
    let split_end = (split_start + split_groups * group).min(k);
    let mut base = split_start;
    while base < split_end {
        #[unroll] for i in 0usize..comptime!((256 * tile_rows).div_ceil(threads)) {
            let t = tid + i * threads; let r = t / 16usize; let col = t % 16usize;
            let mut av = 0.0f32;
            if block_row + r < m && base + col < k {
                let inner = base + col;
                let byte = a[(block_row + r) * a.stride(0) + inner / 2usize * a.stride(1)] as u32;
                av = decode_code((byte >> (u32::cast_from(inner % 2usize) * 4u32)) & 15u32, 14u32);
            }
            left[t] = f16::cast_from(av);
        }
        #[unroll] for i in 0usize..comptime!((256 * tile_cols).div_ceil(threads)) {
            let t = tid + i * threads; let r = t / (16usize * tile_cols); let col = t % (16usize * tile_cols);
            let mut bv = 0.0f32;
            if base + r < k && block_col + col < n {
                let inner = base + r;
                let byte = b[inner / 2usize * b.stride(0) + (block_col + col) * b.stride(1)] as u32;
                bv = decode_code((byte >> (u32::cast_from(inner % 2usize) * 4u32)) & 15u32, 14u32);
            }
            right[t] = f16::cast_from(bv);
        }
        sync_ruda();
        let aa = cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::A, 16usize, 16usize, 16usize,
            cmma::MatrixLayout::RowMajor, &left.slice(warp_row * 256usize, (warp_row + 1usize) * 256usize), 16u32);
        let bb = cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::B, 16usize, 16usize, 16usize,
            cmma::MatrixLayout::RowMajor, &right.slice(warp_col * 16usize, 256usize * tile_cols), (16 * tile_cols) as u32);
        cmma::execute::<f16, f16, f32, f32, _>(&aa, &bb, &acc, &acc); sync_ruda();
        if k - base <= 16usize || (base + 16usize) % group == 0usize {
            cmma::store(&mut partial.slice_mut(warp * 256usize, (warp + 1usize) * 256usize), &acc, 16u32, cmma::MatrixLayout::RowMajor); sync_ruda();
            #[unroll] for i in 0usize..8usize {
                let t = lane + i * 32usize; let row = row_base + t / 16usize; let col = col_base + t % 16usize;
                if row < m && col < n {
                    let mut value = partial[warp * 256usize + t];
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
            else if comptime!(split) { out[RUDA_POS_Z as usize*m*n+row*n+col]=O::cast_from(total[i]); }
            else { crate::scaled_matmul::store(bias, out, row, col, total[i] * alpha, has_bias, output_code); }
        }
    }
}

#[ruda(launch)]
pub(super) fn merge<O:Numeric,P:Float>(partial:&Tensor<f32>,ga:&Tensor<f32>,gb:&Tensor<f32>,bias:&Tensor<P>,out:&mut Tensor<O>,
    #[comptime] splits:usize,#[comptime] global:bool,#[comptime] has_bias:bool,
    #[comptime] output_code:u32,#[define(O)] _output_type:StorageType,#[define(P)] _bias_type:StorageType) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let mut value=0.0f32;
        #[unroll] for s in 0usize..splits {value+=partial[s*out.len()+p];}
        if comptime!(global) {value*=ga[0]*gb[0];}
        let n=out.shape(1);
        crate::scaled_matmul::store(bias,out,p/n,p%n,value,has_bias,output_code);
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
    if super::fp4_mma::try_dispatch(&c,xs,group,global,has_bias) {return;}
    let mma = MmaConfig {a_type: DType::F16.into(), b_type: DType::F16.into(), cd_type: DType::F32.into(), m: 16, n: 16, k: 16};
    assert!(h.plane_size_min == 32 && h.plane_size_max == 32 && h.max_shared_memory_size >= 2048);
    assert!(c.features().matmul.cmma.contains(&mma), "FP4 matmul requires FP16 tensor cores");
    assert!(m <= u32::MAX as usize - 16 && n <= u32::MAX as usize - 16 && k <= u32::MAX as usize - 16);
    let (tile_rows,tile_cols) = if k >= 128 && h.max_units_per_ruda >= 128 && h.max_shared_memory_size >= 8192 {
        if m >= 32 && n >= 32 {(2,2)} else if n >= 64 {(1,4)} else {(1,1)}
    } else {(1,1)};
    let grid_m=m.div_ceil(16*tile_rows);let grid_n=n.div_ceil(16*tile_cols);
    let threads=32*tile_rows*tile_cols;
    assert!(grid_n <= h.max_ruda_count.0 as usize && grid_m <= h.max_ruda_count.1 as usize);
    let tiles=grid_m*grid_n;
    let sms=h.num_streaming_multiprocessors.unwrap_or(108).max(1) as usize;
    let groups=k.div_ceil(group);
    let splits=if k>=2048 && tiles<sms*4 {
        (sms*4).div_ceil(tiles).min(16).min(groups.div_ceil(32)).min((64*1024*1024)/(out.len*4)).max(1)
    } else {1};
    let split_groups=groups.div_ceil(splits);
    if splits>1 {
        let partial=View::packed(c.empty(splits*out.len*4),vec![splits,m,n],0);
        unsafe {
            matmul::launch::<CudaRuntime>(&c,RudaCount::Static(grid_n as u32,grid_m as u32,splits as u32),
                RudaDim::new_1d(threads as u32),a.arg(),b.arg(),sa.arg(),sb.arg(),ga.arg(),gb.arg(),bias.arg(),partial.arg(),
                group,false,false,split_groups,true,tile_rows,tile_cols,0,storage(0),storage(bias.dtype));
            merge::launch::<CudaRuntime>(&c,RudaCount::Static(out.len.div_ceil(128) as u32,1,1),RudaDim::new_1d(128),
                partial.arg(),ga.arg(),gb.arg(),bias.arg(),out.arg(),splits,global,has_bias,out.dtype,storage(out.dtype),storage(bias.dtype));
        }
    } else {
        unsafe { matmul::launch::<CudaRuntime>(&c, RudaCount::Static(grid_n as u32, grid_m as u32, 1),
            RudaDim::new_1d(threads as u32), a.arg(), b.arg(), sa.arg(), sb.arg(), ga.arg(), gb.arg(), bias.arg(), out.arg(),
            group, global, has_bias,groups,false,tile_rows,tile_cols,out.dtype, storage(out.dtype), storage(bias.dtype)); }
    }
    LAUNCHES.fetch_add(if splits>1 {2}else{1}, Ordering::Relaxed); finish_dispatch(&c);
}
