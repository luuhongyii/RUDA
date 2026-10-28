//! FA2-style prefill for A100-class GPUs: 128 query rows x 64 keys per CTA,
//! four warps, each warp owns two m16 query tiles so every K/V fragment feeds
//! two tensor-core MMAs. Q stays in shared memory (reloaded per k16 step), P stays
//! in registers. Interior blocks skip all masking; the softmax scale is folded
//! into a single FFMA before ex2. Causal CTAs run longest-first.
use super::*;

#[ruda(fast_math = FastMath::ReducedPrecision.into())]
fn exp2_fast(value: f32) -> f32 {
    f32::powf(2.0f32, value)
}

#[ruda]
fn stage_rows<F: Float + RudaElement>(source_array: &Array<F>, stage: &mut SharedMemory<F>,
    unit: usize, base_row: usize, row_limit: usize, plane: usize,
    #[comptime] rows: usize, #[comptime] width: usize) {
    #[unroll] for i in 0usize..(rows*width/8/128) {
        let index=(unit+i*128)*8;
        let row=index/width;let col=index%width;
        let mut source=0usize;let mut length=0usize;
        if base_row+row<row_limit {
            source=(plane*row_limit+base_row+row)*width+col;
            length=8;
        }
        let target=row*(width+8)+col;
        barrier::copy_async_checked(&source_array.slice(source,source+length),&mut stage.slice_mut(target,target+8),8u32);
    }
}

#[ruda(launch)]
pub(super) fn flash_wide<F: Float + RudaElement>(
    query: &Array<F>, keys: &Array<F>, values: &Array<F>, output: &mut Array<F>, stats: &mut Array<f32>,
    queries: u32, key_count: u32, heads: u32, kv_heads: u32, scale: f32, scale_log2: f32,
    #[comptime] width: usize, #[comptime] causal: bool,
) {
    let lane=UNIT_POS_X as usize;
    let warp=UNIT_POS_Y as usize;
    let unit=warp*32+lane;
    let head=RUDA_POS_Y as usize;
    let batch=RUDA_POS_Z as usize;
    let mut block_index=RUDA_POS_X as usize;
    if comptime!(causal) {block_index=(queries as usize+127)/128-1-block_index;}
    let query_base=block_index*128;
    let stride=comptime!(width+8);
    let mut qs=SharedMemory::<F>::new_aligned(128*stride,32usize);
    let mut ks=SharedMemory::<F>::new_aligned(64*stride,32usize);
    let mut vs=SharedMemory::<F>::new_aligned(64*stride,32usize);
    let ready=ComptimeOption::<Shared<barrier::Barrier>>::new_Some(barrier::Barrier::shared(128u32,UNIT_POS==0));
    let kv_head=head/(heads/kv_heads) as usize;
    let q_plane=batch*heads as usize+head;
    let kv_plane=batch*kv_heads as usize+kv_head;
    // Q and the first K block arrive together.
    stage_rows(query,&mut qs,unit,query_base,queries as usize,q_plane,128usize,width);
    stage_rows(keys,&mut ks,unit,0usize,key_count as usize,kv_plane,64usize,width);
    ready.as_ref().unwrap().commit_copy_async();
    let def=cmma::MmaDefinition::<F,F,f32>::new(16usize,8usize,16usize);
    let mut accumulators=Sequence::<Array<Vector<f32,Const<2>>>>::new();
    #[unroll] for _ in 0usize..(2*width/8) {
        let mut acc=Array::<Vector<f32,Const<2>>>::new(2usize);
        acc[0]=Vector::new(0.0f32);acc[1]=Vector::new(0.0f32);
        accumulators.push(acc);
    }
    // Four rows per thread: (m tile, row half). Maximum is in raw (unscaled) score units.
    let mut maximum=Array::<f32>::new(4usize);
    let mut denominator=Array::<f32>::new(4usize);
    #[unroll] for r in 0usize..4 {maximum[r]=f32::cast_from(f32::NEG_INFINITY);denominator[r]=0.0f32;}
    let mut end=key_count as usize;
    if comptime!(causal) {end=usize::min(end,query_base+128);}
    for block in 0usize..((end+63)/64) {
        let key_base=block*64;
        ready.as_ref().unwrap().arrive_and_wait();
        // V[i] arrives while QK and the online softmax use only Q and K[i].
        stage_rows(values,&mut vs,unit,key_base,key_count as usize,kv_plane,64usize,width);
        ready.as_ref().unwrap().commit_copy_async();
        let mut scores=Sequence::<Array<Vector<f32,Const<2>>>>::new();
        #[unroll] for _ in 0usize..16 {
            let mut score=Array::<Vector<f32,Const<2>>>::new(2usize);
            score[0]=Vector::new(0.0f32);score[1]=Vector::new(0.0f32);
            scores.push(score);
        }
        #[unroll] for d in 0usize..(width/16) {
            let mut qa=Sequence::<Array<Vector<F,Const<2>>>>::new();
            #[unroll] for mi in 0usize..2 {
                let start=(warp*32+mi*16+lane%16)*stride+d*16+(lane/16)*8;
                qa.push(def.load_matrix::<F,Const<2>>(&qs.slice(start,start+8),cmma::MatrixIdent::A,4usize,false));
            }
            #[unroll] for pair in 0usize..4 {
                let start=((pair*2+lane/16)*8+lane%8)*stride+d*16+(lane/8%2)*8;
                let k4=def.load_matrix::<F,Const<2>>(&ks.slice(start,start+8),cmma::MatrixIdent::B,4usize,false);
                #[unroll] for half in 0usize..2 {
                    let mut key=Array::<Vector<F,Const<2>>>::new(2usize);
                    key[0]=k4[half*2];key[1]=k4[half*2+1];
                    #[unroll] for mi in 0usize..2 {
                        let score=scores.index_mut(mi*8+pair*2+half);
                        def.execute_inplace(&qa[mi],&key,score);
                    }
                }
            }
        }
        // Only the causal diagonal and the sequence tails need element masks.
        let mut full=key_base+64<=key_count as usize && query_base+128<=queries as usize;
        if comptime!(causal) {full=full && key_base+64<=query_base+1;}
        if !full {
            #[unroll] for mi in 0usize..2 {
                #[unroll] for tile in 0usize..8 {
                    let score=scores.index_mut(mi*8+tile);
                    #[unroll] for row in 0usize..2 {
                        let position=query_base+warp*32+mi*16+lane/4+row*8;
                        let mut pair=score[row];
                        #[unroll] for j in 0usize..2 {
                            let column=key_base+tile*8+(lane%4)*2+j;
                            let mut valid=position<queries as usize && column<key_count as usize;
                            if comptime!(causal) {valid=valid && column<=position;}
                            if !valid {pair[j]=f32::cast_from(f32::NEG_INFINITY);}
                        }
                        score[row]=pair;
                    }
                }
            }
        }
        let mut probabilities=Sequence::<Array<Vector<F,Const<2>>>>::new();
        #[unroll] for _ in 0usize..16 {probabilities.push(Array::<Vector<F,Const<2>>>::new(2usize));}
        let mut rescale=Array::<f32>::new(4usize);
        #[unroll] for mi in 0usize..2 {
            #[unroll] for row in 0usize..2 {
                let r=mi*2+row;
                let mut next=maximum[r];
                #[unroll] for tile in 0usize..8 {
                    let pair=scores[mi*8+tile][row];next=f32::max(next,f32::max(pair[0],pair[1]));
                }
                next=f32::max(next,plane_shuffle_xor(next,1u32));
                next=f32::max(next,plane_shuffle_xor(next,2u32));
                if next==f32::cast_from(f32::NEG_INFINITY) {next=0.0f32;}
                let shift=-next*scale_log2;
                let previous_scale=exp2_fast(fma(maximum[r],scale_log2,shift));
                let mut sum=0.0f32;
                #[unroll] for tile in 0usize..8 {
                    let mut pair=scores[mi*8+tile][row];
                    #[unroll] for j in 0usize..2 {pair[j]=exp2_fast(fma(pair[j],scale_log2,shift));sum+=pair[j];}
                    probabilities.index_mut(mi*8+tile)[row]=Vector::cast_from(pair);
                }
                sum+=plane_shuffle_xor(sum,1u32);sum+=plane_shuffle_xor(sum,2u32);
                denominator[r]=fma(denominator[r],previous_scale,sum);
                maximum[r]=next;rescale[r]=previous_scale;
            }
        }
        ready.as_ref().unwrap().arrive_and_wait();
        // Every warp is done with K[i]; fetch K[i+1] while PV consumes V[i].
        if key_base+64<end {
            stage_rows(keys,&mut ks,unit,key_base+64,key_count as usize,kv_plane,64usize,width);
            ready.as_ref().unwrap().commit_copy_async();
        }
        #[unroll] for tile in 0usize..4 {
            let mut probability=Sequence::<Array<Vector<F,Const<2>>>>::new();
            #[unroll] for mi in 0usize..2 {
                let mut p=Array::<Vector<F,Const<2>>>::new(4usize);
                p[0]=probabilities[mi*8+tile*2][0];
                p[1]=probabilities[mi*8+tile*2][1];
                p[2]=probabilities[mi*8+tile*2+1][0];
                p[3]=probabilities[mi*8+tile*2+1][1];
                probability.push(p);
            }
            #[unroll] for d in 0usize..(width/16) {
                let start=(tile*16+lane%16)*stride+(d*2+lane/16)*8;
                let v4=def.load_matrix::<F,Const<2>>(&vs.slice(start,start+8),cmma::MatrixIdent::B,4usize,true);
                #[unroll] for half in 0usize..2 {
                    let mut value=Array::<Vector<F,Const<2>>>::new(2usize);
                    value[0]=v4[half*2];value[1]=v4[half*2+1];
                    #[unroll] for mi in 0usize..2 {
                        let acc=accumulators.index_mut(mi*(width/8)+d*2+half);
                        if comptime!(tile==0) {
                            #[unroll] for row in 0usize..2 {acc[row]=acc[row]*Vector::<f32,Const<2>>::new(rescale[mi*2+row]);}
                        }
                        def.execute_inplace(&probability[mi],&value,acc);
                    }
                }
            }
        }
    }
    #[unroll] for mi in 0usize..2 {
        #[unroll] for row in 0usize..2 {
            let r=mi*2+row;
            let position=query_base+warp*32+mi*16+lane/4+row*8;
            if position<queries as usize {
                let offset=q_plane*queries as usize+position;
                if lane%4==0 {stats[offset*2]=maximum[r]*scale;stats[offset*2+1]=denominator[r];}
                let mut inverse=0.0f32;
                if denominator[r]!=0.0f32 {inverse=1.0f32/denominator[r];}
                #[unroll] for d in 0usize..(width/8) {
                    let pair=accumulators[mi*(width/8)+d][row];
                    #[unroll] for j in 0usize..2 {
                        output[offset*width+d*8+(lane%4)*2+j]=F::cast_from(pair[j]*inverse);
                    }
                }
            }
        }
    }
}
