use super::*;
use ruda_core::{e2m1x2,e4m3,ir::{ElemType,FloatKind,features::{MmaConfig,ScaledMmaConfig}},tensor::DType};
use ruda_kernel::quantization::packed::decode_code;

#[ruda]
fn bytes4(input:&Tensor<u8>,offset:usize,stride:usize,#[comptime] vector:bool)->Vector<u8,Const<4>> {
    let mut value=Vector::<u8,Const<4>>::empty();
    if comptime!(vector) {
        let packed=input.slice(0usize,input.len()).into_vectorized().with_vector_size::<Const<4>>();
        value=packed[offset/4usize];
    } else {
        #[unroll] for j in 0usize..4usize {value[j]=input[offset+j*stride];}
    }
    value
}

#[ruda]
fn signed_bytes(value:Vector<u8,Const<4>>,scale:u8)->Vector<e2m1x2,Const<4>> {
    let mask=(u32::cast_from(scale)>>7u32)*136u32;
    let mut result=Vector::<e2m1x2,Const<4>>::empty();
    #[unroll] for i in 0usize..4usize {result[i]=e2m1x2::reinterpret(u8::cast_from(u32::cast_from(value[i])^mask));}
    result
}

#[ruda]
fn half_pair(value:u8)->Vector<f16,Const<2>> {
    let byte=u32::cast_from(value);let mut pair=Vector::<f16,Const<2>>::empty();
    pair[0usize]=f16::cast_from(decode_code(byte&15u32,14u32));
    pair[1usize]=f16::cast_from(decode_code(byte>>4u32,14u32));
    pair
}

#[ruda]
fn store<O:Numeric,P:Float>(values:&Array<Vector<f32,Const<2>>>,ga:&Tensor<f32>,gb:&Tensor<f32>,bias:&Tensor<P>,out:&mut Tensor<O>,
    row_base:usize,col_base:usize,m:usize,n:usize,#[comptime] split:bool,#[comptime] global:bool,
    #[comptime] has_bias:bool,#[comptime] output_code:u32,#[comptime] acc_offset:usize)
{
    let lane=UNIT_POS_PLANE as usize;let mut alpha=1.0f32;
    if comptime!(global) {alpha=ga[0usize]*gb[0usize];}
    #[unroll] for i in 0usize..2usize {
        let row=row_base+lane/4usize+i*8usize;let value=values[acc_offset+i];
        #[unroll] for j in 0usize..2usize {
            let col=col_base+lane%4usize*2usize+j;
            if row<m && col<n {
                if comptime!(split) {out[RUDA_POS_Z as usize*m*n+row*n+col]=O::cast_from(value[j]);}
                else {crate::scaled_matmul::store(bias,out,row,col,value[j]*alpha,has_bias,output_code);}
            }
        }
    }
}

#[ruda(launch)]
fn native<O:Numeric,P:Float>(a:&Tensor<u8>,b:&Tensor<u8>,sa:&Tensor<u8>,sb:&Tensor<u8>,
    ga:&Tensor<f32>,gb:&Tensor<f32>,bias:&Tensor<P>,out:&mut Tensor<O>,
    #[comptime] tile_rows:usize,#[comptime] tile_cols:usize,#[comptime] column_tiles:usize,#[comptime] split_span:usize,
    #[comptime] split:bool,#[comptime] global:bool,#[comptime] has_bias:bool,
    #[comptime] vector_a:bool,#[comptime] vector_b:bool,#[comptime] output_code:u32,
    #[define(O)] _output_type:StorageType,#[define(P)] _bias_type:StorageType)
{
    let m=a.shape(0);let k=a.shape(1)*2usize;let n=b.shape(1);let warp=UNIT_POS as usize/32usize;
    let lane=UNIT_POS_PLANE as usize;
    let grid_m=m/(16usize*tile_rows)+usize::cast_from(m%(16usize*tile_rows)!=0usize);
    let grid_n=n/(8usize*tile_cols*column_tiles)+usize::cast_from(n%(8usize*tile_cols*column_tiles)!=0usize);
    let position=crate::matmul::tile_position(RUDA_POS_Y as usize*grid_n+RUDA_POS_X as usize,grid_m,grid_n);
    let row_base=position.0*(16usize*tile_rows)+warp/tile_cols*16usize;
    let col_base=(position.1*tile_cols+warp%tile_cols)*(8usize*column_tiles);
    let def=cmma::MmaDefinition::<e2m1x2,e2m1x2,f32>::new_scaled::<e4m3>(16usize,8usize,64usize,4usize);
    let mut a_regs=Array::<Vector<e2m1x2,Const<4>>>::new(4usize);
    let mut b_regs=Array::<Vector<e2m1x2,Const<4>>>::new(2usize);
    let mut acc=Array::<Vector<f32,Const<2>>>::new(2usize*column_tiles);
    #[unroll] for i in 0usize..comptime!(2*column_tiles) {acc[i]=Vector::new(0.0f32);}
    let start=RUDA_POS_Z as usize*split_span;let mut end=start;
    if start<k {end=start+(k-start).min(split_span);}
    let mut base=start;
    while base<end {
        #[unroll] for i in 0usize..4usize {
            let row=row_base+lane/4usize+comptime!(i%2*8);
            let inner=base+lane%4usize*8usize+comptime!(i/2*32);
            let mut data=Vector::new(0u8);let mut factor=0u8;
            if row<m && inner<end {
                data=bytes4(a,row*a.stride(0)+inner/2usize*a.stride(1),a.stride(1),vector_a);
                factor=sa[crate::scaled_matmul::block_index(row,inner,k,16usize)];
            }
            a_regs[i]=signed_bytes(data,factor);
        }
        let scale_row=row_base+lane/4usize+lane%2usize*8usize;
        let mut scales_a=Vector::<e4m3,Const<4>>::empty();
        #[unroll] for i in 0usize..4usize {
            let inner=base+i*16usize;let mut ac=0u8;
            if inner<end && scale_row<m {ac=sa[crate::scaled_matmul::block_index(scale_row,inner,k,16usize)];}
            scales_a[i]=e4m3::reinterpret(ac&127u8);
        }
        #[unroll] for tile in 0usize..column_tiles {
            #[unroll] for i in 0usize..2usize {
                let col=col_base+tile*8usize+lane/4usize;let inner=base+lane%4usize*8usize+i*32usize;
                let mut data=Vector::new(0u8);let mut factor=0u8;
                if col<n && inner<end {
                    data=bytes4(b,inner/2usize*b.stride(0)+col*b.stride(1),b.stride(0),vector_b);
                    factor=sb[crate::scaled_matmul::block_index(col,inner,k,16usize)];
                }
                b_regs[i]=signed_bytes(data,factor);
            }
            let scale_col=col_base+tile*8usize+lane/4usize;
            let mut scales_b=Vector::<e4m3,Const<4>>::empty();
            #[unroll] for i in 0usize..4usize {
                let inner=base+i*16usize;let mut bc=0u8;
                if inner<end && scale_col<n {bc=sb[crate::scaled_matmul::block_index(scale_col,inner,k,16usize)];}
                scales_b[i]=e4m3::reinterpret(bc&127u8);
            }
            let mut current=Array::<Vector<f32,Const<2>>>::new(2usize);
            #[unroll] for i in 0usize..2usize {current[i]=acc[tile*2usize+i];}
            let next=def.execute_scaled(&a_regs,&b_regs,&current,scales_a,scales_b);
            #[unroll] for i in 0usize..2usize {acc[tile*2usize+i]=next[i];}
        }
        base+=64usize;
    }
    #[unroll] for tile in 0usize..column_tiles {
        store(&acc,ga,gb,bias,out,row_base,col_base+tile*8usize,m,n,split,global,has_bias,output_code,comptime!(tile*2));
    }
}

#[ruda(launch)]
fn carrier<O:Numeric,P:Float>(a:&Tensor<u8>,b:&Tensor<u8>,sa:&Tensor<u8>,sb:&Tensor<u8>,
    ga:&Tensor<f32>,gb:&Tensor<f32>,bias:&Tensor<P>,out:&mut Tensor<O>,
    #[comptime] group:usize,#[comptime] tile_rows:usize,#[comptime] tile_cols:usize,#[comptime] column_tiles:usize,#[comptime] split_span:usize,
    #[comptime] split:bool,#[comptime] global:bool,#[comptime] has_bias:bool,#[comptime] output_code:u32,
    #[define(O)] _output_type:StorageType,#[define(P)] _bias_type:StorageType)
{
    let m=a.shape(0);let k=a.shape(1)*2usize;let n=b.shape(1);let warp=UNIT_POS as usize/32usize;
    let lane=UNIT_POS_PLANE as usize;
    let grid_m=m/(16usize*tile_rows)+usize::cast_from(m%(16usize*tile_rows)!=0usize);
    let grid_n=n/(8usize*tile_cols*column_tiles)+usize::cast_from(n%(8usize*tile_cols*column_tiles)!=0usize);
    let position=crate::matmul::tile_position(RUDA_POS_Y as usize*grid_n+RUDA_POS_X as usize,grid_m,grid_n);
    let row_base=position.0*(16usize*tile_rows)+warp/tile_cols*16usize;
    let col_base=(position.1*tile_cols+warp%tile_cols)*(8usize*column_tiles);
    let def=cmma::MmaDefinition::<f16,f16,f32>::new(16usize,8usize,16usize);
    let mut a_regs=Array::<Vector<f16,Const<2>>>::new(4usize);let mut b_regs=Array::<Vector<f16,Const<2>>>::new(2usize);
    let mut acc=Array::<Vector<f32,Const<2>>>::new(2usize*column_tiles);let mut total=Array::<Vector<f32,Const<2>>>::new(2usize*column_tiles);
    #[unroll] for i in 0usize..comptime!(2*column_tiles) {acc[i]=Vector::new(0.0f32);total[i]=Vector::new(0.0f32);}
    let start=RUDA_POS_Z as usize*split_span;let mut end=start;
    if start<k {end=start+(k-start).min(split_span);}
    let mut base=start;
    while base<end {
        #[unroll] for i in 0usize..4usize {
            let row=row_base+lane/4usize+comptime!(i%2*8);let inner=base+lane%4usize*2usize+comptime!(i/2*8);
            let mut data=0u8;
            if row<m && inner<end {data=a[row*a.stride(0)+inner/2usize*a.stride(1)];}
            a_regs[i]=half_pair(data);
        }
        #[unroll] for tile in 0usize..column_tiles {
            #[unroll] for i in 0usize..2usize {
                let col=col_base+tile*8usize+lane/4usize;let inner=base+lane%4usize*2usize+i*8usize;let mut data=0u8;
                if col<n && inner<end {data=b[inner/2usize*b.stride(0)+col*b.stride(1)];}
                b_regs[i]=half_pair(data);
            }
            let mut current=Array::<Vector<f32,Const<2>>>::new(2usize);
            #[unroll] for i in 0usize..2usize {current[i]=acc[tile*2usize+i];}
            def.execute_inplace(&a_regs,&b_regs,&mut current);
            if end-base<=16usize || (base+16usize)%group==0usize {
                #[unroll] for i in 0usize..2usize {
                    let row=row_base+lane/4usize+i*8usize;let partial=current[i];let mut sum=total[tile*2usize+i];
                    #[unroll] for j in 0usize..2usize {
                        let col=col_base+tile*8usize+lane%4usize*2usize+j;
                        if row<m && col<n {
                            let mut value=partial[j];
                            if comptime!(group==32) {value=crate::scaled_matmul::scaled_block(sa,sb,value,row,col,base,k,6u32,6u32);}
                            else {
                                let av=decode_code(u32::cast_from(sa[crate::scaled_matmul::block_index(row,base,k,16usize)]),18u32);
                                let bv=decode_code(u32::cast_from(sb[crate::scaled_matmul::block_index(col,base,k,16usize)]),18u32);
                                value=value*av*bv;
                            }
                            sum[j]+=value;
                        }
                    }
                    total[tile*2usize+i]=sum;current[i]=Vector::new(0.0f32);
                }
            }
            #[unroll] for i in 0usize..2usize {acc[tile*2usize+i]=current[i];}
        }
        base+=16usize;
    }
    #[unroll] for tile in 0usize..column_tiles {
        store(&total,ga,gb,bias,out,row_base,col_base+tile*8usize,m,n,split,global,has_bias,output_code,comptime!(tile*2));
    }
}

pub(super) fn try_dispatch(c:&ComputeClient<CudaRuntime>,xs:&[View],group:usize,global:bool,has_bias:bool)->bool {
    let (a,b,sa,sb,ga,gb,bias,out)=(&xs[0],&xs[1],&xs[2],&xs[3],&xs[4],&xs[5],&xs[6],&xs[7]);
    let (m,k,n)=(a.shape[0],a.shape[1]*2,b.shape[1]);let h=&c.properties().hardware;
    if m==0 || n==0 || k==0 || m>u32::MAX as usize-64 || n>u32::MAX as usize-64 || k>u32::MAX as usize-64
        || h.plane_size_min!=32 || h.plane_size_max!=32 {return false;}
    let packed=StorageType::Packed(ElemType::Float(FloatKind::E2M1),2);
    let native=group==16 && c.features().matmul.scaled_mma.contains(&ScaledMmaConfig{
        a_type:packed,b_type:packed,cd_type:DType::F32.into(),scales_type:ElemType::Float(FloatKind::E4M3).into(),m:16,n:8,k:64,scales_factor:4});
    let manual=c.features().matmul.mma.contains(&MmaConfig{
        a_type:DType::F16.into(),b_type:DType::F16.into(),cd_type:DType::F32.into(),m:16,n:8,k:16});
    if !native && !manual {return false;}
    let (tile_rows,tile_cols)=if h.max_units_per_ruda>=128 && h.max_ruda_dim.0>=128 {
        if m>=32 {(2,2)}else {(1,4)}
    }else {(1,1)};
    let sms=h.num_streaming_multiprocessors.unwrap_or(108).max(1) as usize;
    let grid_m=m.div_ceil(16*tile_rows);
    let column_tiles=if n>=64 && (k>=2048 || grid_m*n.div_ceil(16*tile_cols)>=2*sms) {2}else {1};
    let grid_n=n.div_ceil(8*tile_cols*column_tiles);let tiles=grid_m*grid_n;
    if grid_n>h.max_ruda_count.0 as usize || grid_m>h.max_ruda_count.1 as usize {return false;}
    let quantum=if native {64}else {group};let chunks=k.div_ceil(quantum);
    let splits=if k>=2048 && tiles<sms*4 {(sms*4).div_ceil(tiles).min(16)
        .min(chunks.div_ceil(512/quantum)).min((64*1024*1024)/(out.len*4)).max(1)}else {1};
    let split_span=chunks.div_ceil(splits)*quantum;
    let vector_a=a.strides==[a.shape[1],1] && a.shape[1]%4==0 && a.handle.offset_start.unwrap_or(0)%4==0;
    let vector_b=b.strides==[1,b.shape[0]] && b.shape[0]%4==0 && b.handle.offset_start.unwrap_or(0)%4==0;
    let storage=|code|->StorageType {match code {0=>DType::F32,1=>DType::F16,2=>DType::BF16,_=>unreachable!()}.into()};
    let partial=if splits>1 {Some(View::packed(c.empty(splits*out.len*4),vec![splits,m,n],0))}else {None};
    let target=partial.as_ref().unwrap_or(out);let grid=RudaCount::Static(grid_n as u32,grid_m as u32,splits as u32);
    let threads=RudaDim::new_1d((32*tile_rows*tile_cols) as u32);
    unsafe {
        if native {
            native::launch::<CudaRuntime>(c,grid,threads,a.arg(),b.arg(),sa.arg(),sb.arg(),ga.arg(),gb.arg(),bias.arg(),target.arg(),
                tile_rows,tile_cols,column_tiles,split_span,splits>1,global && splits==1,has_bias && splits==1,vector_a,vector_b,target.dtype,storage(target.dtype),storage(bias.dtype));
        } else {
            carrier::launch::<CudaRuntime>(c,grid,threads,a.arg(),b.arg(),sa.arg(),sb.arg(),ga.arg(),gb.arg(),bias.arg(),target.arg(),
                group,tile_rows,tile_cols,column_tiles,split_span,splits>1,global && splits==1,has_bias && splits==1,target.dtype,storage(target.dtype),storage(bias.dtype));
        }
        if let Some(partial)=&partial {
            crate::fp4_matmul::merge::launch::<CudaRuntime>(c,RudaCount::Static(out.len.div_ceil(128) as u32,1,1),RudaDim::new_1d(128),
                partial.arg(),ga.arg(),gb.arg(),bias.arg(),out.arg(),splits,global,has_bias,out.dtype,storage(out.dtype),storage(bias.dtype));
        }
    }
    LAUNCHES.fetch_add(if splits>1 {2}else {1},Ordering::Relaxed);finish_dispatch(c);true
}
