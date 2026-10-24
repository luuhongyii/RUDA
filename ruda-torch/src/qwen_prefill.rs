use super::*;

#[ruda(launch)]
pub(super) fn flash<F: Float + RudaElement>(
    query: &Array<F>, keys: &Array<F>, values: &Array<F>, output: &mut Array<F>,
    queries: u32, key_count: u32, heads: u32, kv_heads: u32, scale: f32,
    #[comptime] width: usize, #[comptime] query_block: usize, #[comptime] causal: bool,
) {
    let lane = UNIT_POS_X as usize;
    let warp = UNIT_POS_Y as usize;
    let unit = warp * 32 + lane;
    let head = RUDA_POS_Y as usize;
    let batch = RUDA_POS_Z as usize;
    let query_base = RUDA_POS_X as usize * query_block;
    let threads = comptime!(query_block * 2);
    let qstride = comptime!(width + 8);
    let temporary_stride = comptime!(usize::max(width + 8,72));
    let mut query_stage = SharedMemory::<F>::new_aligned(query_block * qstride,32usize);
    let mut key_stage = SharedMemory::<F>::new_aligned(64 * qstride,32usize);
    let mut value_stage = SharedMemory::<F>::new_aligned(64 * qstride,32usize);
    let mut probability = SharedMemory::<F>::new_aligned(query_block * 72usize,32usize);
    let mut temporary = SharedMemory::<f32>::new_aligned(query_block * temporary_stride,32usize);
    let mut row_scale = SharedMemory::<f32>::new(query_block);
    #[unroll] for i in 0usize..((query_block * width + threads - 1) / threads) {
        let t = unit + i * threads;
        if t < query_block * width {
            let row = t / width;
            let column = t % width;
            let mut value = F::cast_from(0.0f32);
            if query_base + row < queries as usize {
                value = query[((batch * heads as usize + head) * queries as usize + query_base + row) * width + column];
            }
            query_stage[row * qstride + column] = value;
        }
    }
    sync_ruda();
    let qbegin = warp * 16 * qstride;
    let pbegin = warp * 16 * 72;
    let tbegin = warp * 16 * temporary_stride;
    let mut qfragments = Sequence::<cmma::Matrix<F>>::new();
    #[unroll] for d in 0usize..(width / 16) {
        qfragments.push(cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::A,16usize,16usize,16usize,
            cmma::MatrixLayout::RowMajor,&query_stage.slice(qbegin + d * 16,qbegin + 16 * qstride),qstride as u32));
    }
    let mut accumulator = Sequence::<RuntimeCell<f32>>::new();
    #[unroll] for _ in 0usize..(width / 2) { accumulator.push(RuntimeCell::new(0.0f32)); }
    let mut maximum = Sequence::<RuntimeCell<f32>>::new();
    let mut denominator = Sequence::<RuntimeCell<f32>>::new();
    #[unroll] for _ in 0usize..4 {
        maximum.push(RuntimeCell::new(f32::cast_from(f32::NEG_INFINITY)));
        denominator.push(RuntimeCell::new(0.0f32));
    }
    let kv_head = head / (heads / kv_heads) as usize;
    let mut end = key_count as usize;
    if comptime!(causal) { end = usize::min(end,query_base + query_block); }
    let blocks = (end + 63) / 64;
    for block in 0usize..blocks {
        let key_base = block * 64;
        #[unroll] for i in 0usize..((64 * width + threads - 1) / threads) {
            let t = unit + i * threads;
            if t < 64 * width {
                let row = t / width;
                let column = t % width;
                let mut key = F::cast_from(0.0f32);
                let mut value = F::cast_from(0.0f32);
                if key_base + row < key_count as usize {
                    let source = ((batch * kv_heads as usize + kv_head) * key_count as usize + key_base + row) * width + column;
                    key = keys[source]; value = values[source];
                }
                key_stage[row * qstride + column] = key;
                value_stage[row * qstride + column] = value;
            }
        }
        sync_ruda();
        #[unroll] for tile in 0usize..4 {
            let score = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,16usize,16usize,16usize,
                cmma::MatrixLayout::Undefined,0.0f32);
            #[unroll] for d in 0usize..(width / 16) {
                let key = cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::B,16usize,16usize,16usize,
                    cmma::MatrixLayout::ColMajor,&key_stage.slice(tile * 16 * qstride + d * 16,64 * qstride),qstride as u32);
                cmma::execute::<F,F,f32,f32,_>(&qfragments[d],&key,&score,&score);
            }
            cmma::store(&mut temporary.slice_mut(tbegin + tile * 16,tbegin + 16 * temporary_stride),
                &score,temporary_stride as u32,cmma::MatrixLayout::RowMajor);
        }
        sync_ruda();
        #[unroll] for group in 0usize..4 {
            let row = group * 4 + lane / 8;
            let position = query_base + warp * 16 + row;
            let mut scores = Vector::<f32,Const<8>>::new(0.0f32);
            let mut next = f32::cast_from(f32::NEG_INFINITY);
            #[unroll] for j in 0usize..8 {
                let column = lane % 8 + j * 8;
                let mut score = f32::cast_from(f32::NEG_INFINITY);
                if position < queries as usize && key_base + column < key_count as usize {
                    if !comptime!(causal) || key_base + column <= position {
                        score = temporary[tbegin + row * temporary_stride + column] * scale;
                    }
                }
                scores[j] = score; next = f32::max(next,score);
            }
            next = f32::max(next,plane_shuffle_xor(next,1u32));
            next = f32::max(next,plane_shuffle_xor(next,2u32));
            next = f32::max(next,plane_shuffle_xor(next,4u32));
            next = f32::max(next,maximum[group].read());
            if next == f32::cast_from(f32::NEG_INFINITY) { next = 0.0f32; }
            let mut previous_scale = 0.0f32;
            if denominator[group].read() != 0.0f32 { previous_scale = (maximum[group].read() - next).exp(); }
            let mut sum = 0.0f32;
            #[unroll] for j in 0usize..8 {
                let value = (scores[j] - next).exp();
                probability[pbegin + row * 72 + lane % 8 + j * 8] = F::cast_from(value);
                sum += value;
            }
            sum += plane_shuffle_xor(sum,1u32);
            sum += plane_shuffle_xor(sum,2u32);
            sum += plane_shuffle_xor(sum,4u32);
            denominator[group].store(fma(denominator[group].read(),previous_scale,sum));
            maximum[group].store(next);
            if lane % 8 == 0 { row_scale[warp * 16 + row] = previous_scale; }
        }
        sync_ruda();
        #[unroll] for d in 0usize..(width / 16) {
            let product = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,16usize,16usize,16usize,
                cmma::MatrixLayout::Undefined,0.0f32);
            #[unroll] for tile in 0usize..4 {
                let left = cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::A,16usize,16usize,16usize,
                    cmma::MatrixLayout::RowMajor,&probability.slice(pbegin + tile * 16,pbegin + 16 * 72),72u32);
                let right = cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::B,16usize,16usize,16usize,
                    cmma::MatrixLayout::RowMajor,&value_stage.slice(tile * 16 * qstride + d * 16,64 * qstride),qstride as u32);
                cmma::execute::<F,F,f32,f32,_>(&left,&right,&product,&product);
            }
            cmma::store(&mut temporary.slice_mut(tbegin + d * 16,tbegin + 16 * temporary_stride),
                &product,temporary_stride as u32,cmma::MatrixLayout::RowMajor);
        }
        sync_ruda();
        #[unroll] for j in 0usize..(width / 2) {
            let index = lane + j * 32;
            let row = index / width;
            accumulator[j].store(fma(accumulator[j].read(),row_scale[warp * 16 + row],
                temporary[tbegin + row * temporary_stride + index % width]));
        }
        sync_ruda();
    }
    #[unroll] for group in 0usize..4 {
        if lane % 8 == 0 { row_scale[warp * 16 + group * 4 + lane / 8] = denominator[group].read(); }
    }
    sync_ruda();
    #[unroll] for j in 0usize..(width / 2) {
        let index = lane + j * 32;
        let row = index / width;
        let position = query_base + warp * 16 + row;
        if position < queries as usize {
            output[((batch * heads as usize + head) * queries as usize + position) * width + index % width] =
                F::cast_from(accumulator[j].read() / row_scale[warp * 16 + row]);
        }
    }
}

#[ruda(launch)]
pub(super) fn repeat_heads<F: Float + RudaElement>(input:&Array<F>,output:&mut Array<F>,heads:u32,kv_heads:u32,span:u32) {
    let index = ABSOLUTE_POS as usize;
    if index < output.len() {
        let head = index / span as usize % heads as usize;
        let batch = index / span as usize / heads as usize;
        output[index] = input[(batch * kv_heads as usize + head / (heads / kv_heads) as usize) * span as usize + index % span as usize];
    }
}
