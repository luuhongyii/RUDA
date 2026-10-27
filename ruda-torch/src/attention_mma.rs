use super::*;

#[ruda(fast_math = FastMath::ReducedPrecision.into())]
fn softmax_exp(value: f32) -> f32 {
    f32::powf(2.0f32, value * core::f32::consts::LOG2_E)
}

#[ruda(launch)]
pub(super) fn flash<F: Float + RudaElement>(
    query: &Array<F>, keys: &Array<F>, values: &Array<F>, output: &mut Array<F>,
    mask: &Tensor<f32>, stats: &mut Array<f32>,
    queries: u32, key_count: u32, heads: u32, kv_heads: u32, scale: f32,
    #[comptime] width: usize, #[comptime] causal: bool, #[comptime] mask_kind: u32,
    #[comptime] asynchronous: bool,
) {
    let lane=UNIT_POS_X as usize;
    let warp=UNIT_POS_Y as usize;
    let unit=warp*32+lane;
    let head=RUDA_POS_Y as usize;
    let batch=RUDA_POS_Z as usize;
    let query_base=RUDA_POS_X as usize*64;
    let stride=comptime!(width+8);
    let mut ks=SharedMemory::<F>::new_aligned(64*stride,32usize);
    let mut vs=SharedMemory::<F>::new_aligned(64*stride,32usize);
    let ready=if comptime!(asynchronous) {
        ComptimeOption::<Shared<barrier::Barrier>>::new_Some(barrier::Barrier::shared(128u32,UNIT_POS==0))
    } else {ComptimeOption::<Shared<barrier::Barrier>>::new_None()};
    if comptime!(asynchronous) {
        #[unroll] for i in 0usize..(width/16) {
            let index=(unit+i*128)*8;
            let row=index/width;let col=index%width;
            let mut source=0usize;let mut length=0usize;
            if query_base+row<queries as usize {
                source=((batch*heads as usize+head)*queries as usize+query_base+row)*width+col;
                length=8;
            }
            let target=row*stride+col;
            barrier::copy_async_checked(&query.slice(source,source+length),&mut ks.slice_mut(target,target+8),8u32);
        }
        ready.as_ref().unwrap().commit_copy_async();
        ready.as_ref().unwrap().arrive_and_wait();
    } else {
        #[unroll] for i in 0usize..(width/2) {
            let index=unit+i*128;
            let row=index/width;
            let col=index%width;
            let mut value=F::cast_from(0.0f32);
            if query_base+row<queries as usize {
                value=query[((batch*heads as usize+head)*queries as usize+query_base+row)*width+col];
            }
            ks[row*stride+col]=value;
        }
        sync_ruda();
    }
    let def=cmma::MmaDefinition::<F,F,f32>::new(16usize,8usize,16usize);
    let mut qfragments=Sequence::<Array<Vector<F,Const<2>>>>::new();
    #[unroll] for d in 0usize..(width/16) {
        let start=(warp*16+lane%16)*stride+d*16+(lane/16)*8;
        qfragments.push(def.load_matrix::<F,Const<2>>(&ks.slice(start,start+8),cmma::MatrixIdent::A,4usize,false));
    }
    // All Q fragments must be resident before any warp reuses this stage for K.
    sync_ruda();
    let mut accumulators=Sequence::<Array<Vector<f32,Const<2>>>>::new();
    #[unroll] for _ in 0usize..(width/8) {
        let mut acc=Array::<Vector<f32,Const<2>>>::new(2usize);
        acc[0]=Vector::new(0.0f32);acc[1]=Vector::new(0.0f32);
        accumulators.push(acc);
    }
    let mut maximum=Vector::<f32,Const<2>>::new(f32::cast_from(f32::NEG_INFINITY));
    let mut denominator=Vector::<f32,Const<2>>::new(0.0f32);
    let kv_head=head/(heads/kv_heads) as usize;
    let mut end=key_count as usize;
    if comptime!(causal) {end=usize::min(end,query_base+64);}
    for block in 0usize..((end+63)/64) {
        let key_base=block*64;
        if comptime!(asynchronous) {
            #[unroll] for i in 0usize..(width/16) {
                let index=(unit+i*128)*8;
                let row=index/width;let col=index%width;
                let mut source=0usize;let mut length=0usize;
                if key_base+row<key_count as usize {
                    source=((batch*kv_heads as usize+kv_head)*key_count as usize+key_base+row)*width+col;
                    length=8;
                }
                let target=row*stride+col;
                barrier::copy_async_checked(&keys.slice(source,source+length),&mut ks.slice_mut(target,target+8),8u32);
            }
            ready.as_ref().unwrap().commit_copy_async();
            ready.as_ref().unwrap().arrive_and_wait();
            // V can arrive while QK and online softmax execute using only K.
            #[unroll] for i in 0usize..(width/16) {
                let index=(unit+i*128)*8;
                let row=index/width;let col=index%width;
                let mut source=0usize;let mut length=0usize;
                if key_base+row<key_count as usize {
                    source=((batch*kv_heads as usize+kv_head)*key_count as usize+key_base+row)*width+col;
                    length=8;
                }
                let target=row*stride+col;
                barrier::copy_async_checked(&values.slice(source,source+length),&mut vs.slice_mut(target,target+8),8u32);
            }
            ready.as_ref().unwrap().commit_copy_async();
        } else {
            #[unroll] for i in 0usize..(width/2) {
                let index=unit+i*128;
                let row=index/width;let col=index%width;
                let mut key=F::cast_from(0.0f32);let mut value=F::cast_from(0.0f32);
                if key_base+row<key_count as usize {
                    let source=((batch*kv_heads as usize+kv_head)*key_count as usize+key_base+row)*width+col;
                    key=keys[source];value=values[source];
                }
                ks[row*stride+col]=key;vs[row*stride+col]=value;
            }
            sync_ruda();
        }
        let mut scores=Sequence::<Array<Vector<f32,Const<2>>>>::new();
        #[unroll] for tile in 0usize..8 {
            let mut score=Array::<Vector<f32,Const<2>>>::new(2usize);
            score[0]=Vector::new(0.0f32);score[1]=Vector::new(0.0f32);
            #[unroll] for d in 0usize..(width/16) {
                let start=(tile*8+lane%8)*stride+d*16+(lane/8%2)*8;
                let key=def.load_matrix::<F,Const<2>>(&ks.slice(start,start+8),cmma::MatrixIdent::B,2usize,false);
                def.execute_inplace(&qfragments[d],&key,&mut score);
            }
            #[unroll] for row in 0usize..2 {
                let position=query_base+warp*16+lane/4+row*8;
                let mut pair=score[row];
                #[unroll] for j in 0usize..2 {
                    let column=key_base+tile*8+(lane%4)*2+j;
                    let mut value=f32::cast_from(f32::NEG_INFINITY);
                    if position<queries as usize && column<key_count as usize {
                        if !comptime!(causal) || column<=position {
                            value=pair[j]*scale;
                            if comptime!(mask_kind!=0) {
                                let mut remaining=((batch*heads as usize+head)*queries as usize+position)*key_count as usize+column;
                                let mut offset=0usize;let mut axis=mask.rank();
                                while axis>0 {axis-=1;offset+=(remaining%mask.shape(axis))*mask.stride(axis);remaining/=mask.shape(axis);}
                                let m=mask[offset];
                                if comptime!(mask_kind==1) {if m==0.0f32 {value=f32::cast_from(f32::NEG_INFINITY);}}
                                else {value+=m;}
                            }
                        }
                    }
                    pair[j]=value;
                }
                score[row]=pair;
            }
            scores.push(score);
        }
        let mut rescale=Vector::<f32,Const<2>>::new(0.0f32);
        #[unroll] for row in 0usize..2 {
            let mut next=maximum[row];
            #[unroll] for tile in 0usize..8 {
                let pair=scores[tile][row];next=f32::max(next,f32::max(pair[0],pair[1]));
            }
            next=f32::max(next,plane_shuffle_xor(next,1u32));
            next=f32::max(next,plane_shuffle_xor(next,2u32));
            if next==f32::cast_from(f32::NEG_INFINITY) {next=0.0f32;}
            let mut previous_scale=0.0f32;
            if denominator[row]!=0.0f32 {previous_scale=softmax_exp(maximum[row]-next);}
            let mut sum=0.0f32;
            #[unroll] for tile in 0usize..8 {
                let mut pair=scores[tile][row];
                #[unroll] for j in 0usize..2 {pair[j]=softmax_exp(pair[j]-next);sum+=pair[j];}
                let score=scores.index_mut(tile);
                score[row]=pair;
            }
            sum+=plane_shuffle_xor(sum,1u32);sum+=plane_shuffle_xor(sum,2u32);
            denominator[row]=fma(denominator[row],previous_scale,sum);
            maximum[row]=next;rescale[row]=previous_scale;
        }
        #[unroll] for d in 0usize..(width/8) {
            let acc=accumulators.index_mut(d);
            #[unroll] for row in 0usize..2 {acc[row]=acc[row]*Vector::<f32,Const<2>>::new(rescale[row]);}
        }
        if comptime!(asynchronous) { ready.as_ref().unwrap().arrive_and_wait(); }
        #[unroll] for tile in 0usize..4 {
            let mut probability=Array::<Vector<F,Const<2>>>::new(4usize);
            probability[0]=Vector::cast_from(scores[tile*2][0]);
            probability[1]=Vector::cast_from(scores[tile*2][1]);
            probability[2]=Vector::cast_from(scores[tile*2+1][0]);
            probability[3]=Vector::cast_from(scores[tile*2+1][1]);
            #[unroll] for d in 0usize..(width/8) {
                let start=(tile*16+lane%16)*stride+d*8;
                let value=def.load_matrix::<F,Const<2>>(&vs.slice(start,start+8),cmma::MatrixIdent::B,2usize,true);
                let acc=accumulators.index_mut(d);
                def.execute_inplace(&probability,&value,acc);
            }
        }
        sync_ruda();
    }
    #[unroll] for row in 0usize..2 {
        let position=query_base+warp*16+lane/4+row*8;
        if position<queries as usize {
            let offset=((batch*heads as usize+head)*queries as usize+position);
            if lane%4==0 {stats[offset*2]=maximum[row];stats[offset*2+1]=denominator[row];}
            #[unroll] for d in 0usize..(width/8) {
                let pair=accumulators[d][row];
                #[unroll] for j in 0usize..2 {
                    let mut value=pair[j];if denominator[row]!=0.0f32 {value/=denominator[row];}
                    output[offset*width+d*8+(lane%4)*2+j]=F::cast_from(value);
                }
            }
        }
    }
}
