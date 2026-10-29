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
    if base_row+rows<=row_limit {
        let base=(plane*row_limit+base_row)*width;
        #[unroll] for i in 0usize..(rows*width/8/128) {
            let index=(unit+i*128)*8;
            let target=index/width*(width+8)+index%width;
            barrier::copy_async(&source_array.slice(base+index,base+index+8),&mut stage.slice_mut(target,target+8),8u32);
        }
    } else {
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
}

#[ruda(launch)]
pub(super) fn flash_wide<F: Float + RudaElement>(
    query: &Array<F>, keys: &Array<F>, values: &Array<F>, output: &mut Array<Vector<F,Const<2>>>, stats: &mut Array<f32>,
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
    let kv_head=head/(heads/kv_heads) as usize;
    let q_plane=batch*heads as usize+head;
    let kv_plane=batch*kv_heads as usize+kv_head;
    // Q and the first K block arrive together.
    stage_rows(query,&mut qs,unit,query_base,queries as usize,q_plane,128usize,width);
    stage_rows(keys,&mut ks,unit,0usize,key_count as usize,kv_plane,64usize,width);
    barrier::commit_copy_group();
    let def=cmma::MmaDefinition::<F,F,f32>::new(16usize,8usize,16usize);
    let mut accumulators=Sequence::<Vector<f32,Const<4>>>::new();
    #[unroll] for _ in 0usize..(2*width/8) {
        accumulators.push(Vector::new(0.0f32));
    }
    // Four rows per thread: (m tile, row half). Maximum is in raw (unscaled) score units.
    let mut maximum=Array::<f32>::new(4usize);
    let mut denominator=Array::<f32>::new(4usize);
    #[unroll] for r in 0usize..4 {maximum[r]=f32::cast_from(f32::NEG_INFINITY);denominator[r]=0.0f32;}
    let mut end=key_count as usize;
    if comptime!(causal) {end=usize::min(end,query_base+128);}
    for block in 0usize..((end+63)/64) {
        let key_base=block*64;
        barrier::wait_copy_group(0u32);
        sync_ruda();
        // V[i] arrives while QK and the online softmax use only Q and K[i].
        stage_rows(values,&mut vs,unit,key_base,key_count as usize,kv_plane,64usize,width);
        barrier::commit_copy_group();
        let mut scores=Sequence::<Vector<f32,Const<4>>>::new();
        #[unroll] for _ in 0usize..16 {
            scores.push(Vector::new(0.0f32));
        }
        #[unroll] for d in 0usize..(width/16) {
            let mut qa=Sequence::<Vector<F,Const<8>>>::new();
            #[unroll] for mi in 0usize..2 {
                let start=(warp*32+mi*16+lane%16)*stride+d*16+(lane/16)*8;
                qa.push(def.load_matrix_vector::<F,Const<8>>(&qs.slice(start,start+8),4usize,false));
            }
            #[unroll] for pair in 0usize..4 {
                let start=((pair*2+lane/16)*8+lane%8)*stride+d*16+(lane/8%2)*8;
                let k4=def.load_matrix_vector::<F,Const<8>>(&ks.slice(start,start+8),4usize,false);
                #[unroll] for half in 0usize..2 {
                    let mut key=Vector::<F,Const<4>>::empty();
                    #[unroll] for j in 0usize..4 {key[j]=k4[half*4+j];}
                    #[unroll] for mi in 0usize..2 {
                        let score=scores.index_mut(mi*8+pair*2+half);
                        def.execute_vectors_inplace(&qa[mi],&key,score);
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
                        #[unroll] for j in 0usize..2 {
                            let column=key_base+tile*8+(lane%4)*2+j;
                            let mut valid=position<queries as usize && column<key_count as usize;
                            if comptime!(causal) {valid=valid && column<=position;}
                            if !valid {score[row*2+j]=f32::cast_from(f32::NEG_INFINITY);}
                        }
                    }
                }
            }
        }
        let mut probabilities=Sequence::<Vector<F,Const<4>>>::new();
        #[unroll] for _ in 0usize..16 {probabilities.push(Vector::empty());}
        let mut rescale=Array::<f32>::new(4usize);
        #[unroll] for mi in 0usize..2 {
            #[unroll] for row in 0usize..2 {
                let r=mi*2+row;
                let mut next=maximum[r];
                #[unroll] for tile in 0usize..8 {
                    let score=scores[mi*8+tile];next=f32::max(next,f32::max(score[row*2],score[row*2+1]));
                }
                next=f32::max(next,plane_shuffle_xor(next,1u32));
                next=f32::max(next,plane_shuffle_xor(next,2u32));
                if next==f32::cast_from(f32::NEG_INFINITY) {next=0.0f32;}
                let shift=-next*scale_log2;
                let previous_scale=exp2_fast(fma(maximum[r],scale_log2,shift));
                let mut sum=0.0f32;
                #[unroll] for tile in 0usize..8 {
                    let score=scores[mi*8+tile];
                    #[unroll] for j in 0usize..2 {
                        let value=exp2_fast(fma(score[row*2+j],scale_log2,shift));sum+=value;
                        probabilities.index_mut(mi*8+tile)[row*2+j]=F::cast_from(value);
                    }
                }
                sum+=plane_shuffle_xor(sum,1u32);sum+=plane_shuffle_xor(sum,2u32);
                denominator[r]=fma(denominator[r],previous_scale,sum);
                maximum[r]=next;rescale[r]=previous_scale;
            }
        }
        barrier::wait_copy_group(0u32);
        sync_ruda();
        // Every warp is done with K[i]; fetch K[i+1] while PV consumes V[i].
        if key_base+64<end {
            stage_rows(keys,&mut ks,unit,key_base+64,key_count as usize,kv_plane,64usize,width);
            barrier::commit_copy_group();
        }
        #[unroll] for tile in 0usize..4 {
            let mut probability=Sequence::<Vector<F,Const<8>>>::new();
            #[unroll] for mi in 0usize..2 {
                let mut p=Vector::<F,Const<8>>::empty();
                #[unroll] for j in 0usize..4 {
                    p[j]=probabilities[mi*8+tile*2][j];
                    p[4+j]=probabilities[mi*8+tile*2+1][j];
                }
                probability.push(p);
            }
            #[unroll] for d in 0usize..(width/16) {
                let start=(tile*16+lane%16)*stride+(d*2+lane/16)*8;
                let v4=def.load_matrix_vector::<F,Const<8>>(&vs.slice(start,start+8),4usize,true);
                #[unroll] for half in 0usize..2 {
                    let mut value=Vector::<F,Const<4>>::empty();
                    #[unroll] for j in 0usize..4 {value[j]=v4[half*4+j];}
                    #[unroll] for mi in 0usize..2 {
                        let acc=accumulators.index_mut(mi*(width/8)+d*2+half);
                        if comptime!(tile==0) {
                            #[unroll] for row in 0usize..2 {
                                #[unroll] for j in 0usize..2 {acc[row*2+j]=acc[row*2+j]*rescale[mi*2+row];}
                            }
                        }
                        def.execute_vectors_inplace(&probability[mi],&value,acc);
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
                    let acc=accumulators[mi*(width/8)+d];
                    let mut result=Vector::<F,Const<2>>::empty();
                    #[unroll] for j in 0usize..2 {
                        result[j]=F::cast_from(acc[row*2+j]*inverse);
                    }
                    output[offset*(width/2)+d*4+lane%4]=result;
                }
            }
        }
    }
}
