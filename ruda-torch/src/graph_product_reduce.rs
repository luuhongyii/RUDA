use super::*;

#[ruda]
fn product<F:Float>(left:&Tensor<F>,right:&Tensor<F>,position:usize,#[comptime] shape:Vec<usize>)->f32 {
    let a=f32::cast_from(left[crate::graph_fusion::address(left,position,shape.clone())]);
    let b=f32::cast_from(right[crate::graph_fusion::address(right,position,shape)]);
    f32::cast_from(F::cast_from(a*b))
}

#[ruda]
fn product_vector<F:Float,V:Size>(left:&Tensor<F>,right:&Tensor<F>,position:usize,
    #[comptime] shape:Vec<usize>,#[comptime] left_vector:bool,#[comptime] right_vector:bool)->Vector<f32,V>
{
    let a=crate::graph_fusion::address(left,position,shape.clone());
    let b=crate::graph_fusion::address(right,position,shape);
    let mut x=Vector::<F,V>::new(F::cast_from(0));let mut y=Vector::<F,V>::new(F::cast_from(0));
    if comptime!(left_vector) {
        let packed=left.slice(0usize,left.len()).into_vectorized().with_vector_size::<V>();x=packed[a/V::value()];
    } else {x=Vector::<F,V>::new(left[a]);}
    if comptime!(right_vector) {
        let packed=right.slice(0usize,right.len()).into_vectorized().with_vector_size::<V>();y=packed[b/V::value()];
    } else {y=Vector::<F,V>::new(right[b]);}
    let mut result=Vector::<f32,V>::new(0.0f32);
    #[unroll] for lane in 0usize..V::value() {result[lane]=f32::cast_from(F::cast_from(f32::cast_from(x[lane])*f32::cast_from(y[lane])));}
    result
}

#[ruda(launch)]
fn suffix<F:Float,V:Size>(left:&Tensor<F>,right:&Tensor<F>,out:&mut Array<F>,
    #[comptime] shape:Vec<usize>,#[comptime] width:usize,#[comptime] threads:usize,
    #[comptime] left_vector:bool,#[comptime] right_vector:bool,#[comptime] mean:bool)
{
    let mut row=RUDA_POS_X as usize;let lane=UNIT_POS as usize%threads;
    if comptime!(threads==32) {row=row*4+UNIT_POS as usize/32;}
    if row<out.len() {
        let mut local=Vector::<f32,V>::new(0.0f32);let mut column=lane;
        while column<width/V::value() {
            local+=product_vector::<F,V>(left,right,row*width+column*V::value(),shape.clone(),left_vector,right_vector);
            column+=threads;
        }
        let mut value=0.0f32;#[unroll] for i in 0usize..V::value() {value+=local[i];}
        let mut sum=plane_sum(value);
        if comptime!(threads>32) {
            let mut partials=SharedMemory::<f32>::new(threads/32);
            if lane%32==0 {partials[lane/32]=sum;}sync_ruda();
            let mut v=0.0f32;if lane%32<threads/32 {v=partials[lane%32];}sum=plane_sum(v);
        }
        if comptime!(mean) {sum/=width as f32;}
        if lane==0 {out[row]=F::cast_from(sum);}
    }
}

#[ruda(launch)]
fn columns<F:Float,V:Size>(left:&Tensor<F>,right:&Tensor<F>,out:&mut Array<F>,
    #[comptime] shape:Vec<usize>,#[comptime] width:usize,#[comptime] inner:usize,#[comptime] mean:bool,
    #[comptime] left_vector:bool,#[comptime] right_vector:bool)
{
    let tiles=comptime!(inner.div_ceil(32*V::value()));let outer=RUDA_POS_X as usize/tiles;
    let lane=UNIT_POS_X as usize;let part=UNIT_POS_Y as usize;
    let column=((RUDA_POS_X as usize%tiles)*32+lane)*V::value();let mut sum=Vector::<f32,V>::new(0.0f32);
    if column<inner {
        let mut row=part;
        while row<width {
            sum+=product_vector::<F,V>(left,right,(outer*width+row)*inner+column,shape.clone(),left_vector,right_vector);
            row+=8usize;
        }
    }
    let mut partials=SharedMemory::<Vector<f32,V>>::new(256usize);partials[part*32+lane]=sum;sync_ruda();
    if part==0 && column<inner {
        let mut value=partials[lane];#[unroll] for p in 1usize..8 {value+=partials[p*32+lane];}
        if comptime!(mean) {value/=Vector::new(width as f32);}
        let mut packed=out.slice_mut(0usize,out.len()).into_vectorized().with_vector_size::<V>();
        packed[(outer*inner+column)/V::value()]=Vector::<F,V>::cast_from(value);
    }
}

#[ruda(launch)]
fn arbitrary<F:Float>(left:&Tensor<F>,right:&Tensor<F>,out:&mut Array<F>,
    #[comptime] shape:Vec<usize>,#[comptime] strides:Vec<usize>,#[comptime] mask:u32,
    #[comptime] width:usize,#[comptime] mean:bool)
{
    let row=RUDA_POS_X as usize;let lane=UNIT_POS as usize;
    let mut remaining=row;let mut base=0usize;
    #[unroll] for reverse in 0usize..shape.len() {
        let dim=comptime!(shape.len()-1-reverse);
        if comptime!(mask&(1u32<<dim)==0) {
            base+=(remaining%comptime!(shape[dim]))*comptime!(strides[dim]);remaining/=comptime!(shape[dim]);
        }
    }
    let mut sum=0.0f32;let mut column=lane;
    while column<width {
        let mut index=column;let mut position=base;
        #[unroll] for reverse in 0usize..shape.len() {
            let dim=comptime!(shape.len()-1-reverse);
            if comptime!(mask&(1u32<<dim)!=0) {
                position+=(index%comptime!(shape[dim]))*comptime!(strides[dim]);index/=comptime!(shape[dim]);
            }
        }
        sum+=product(left,right,position,shape.clone());column+=256usize;
    }
    sum=plane_sum(sum);let mut partials=SharedMemory::<f32>::new(8usize);
    if lane%32==0 {partials[lane/32]=sum;}sync_ruda();
    if lane<32 {
        let mut value=0.0f32;if lane<8 {value=partials[lane];}value=plane_sum(value);
        if comptime!(mean) {value/=width as f32;}
        if lane==0 {out[row]=F::cast_from(value);}
    }
}

#[ruda(launch)]
fn contraction<F:Float>(left:&Array<F>,right:&Array<F>,out:&mut Array<F>,
    #[comptime] m:usize,#[comptime] n:usize,#[comptime] k:usize,
    #[comptime] batch_shape:Vec<usize>,#[comptime] batch_left:Vec<usize>,
    #[comptime] batch_right:Vec<usize>,#[comptime] batch_out:Vec<usize>,
    #[comptime] am:usize,#[comptime] ak:usize,#[comptime] bn:usize,#[comptime] bk:usize,
    #[comptime] om:usize,#[comptime] on:usize,#[comptime] mean:bool)
{
    let tiles=comptime!(m.div_ceil(16)*n.div_ceil(32));
    let block=RUDA_POS_X as usize%tiles;let mut batch=RUDA_POS_X as usize/tiles;
    let row=(block/comptime!(n.div_ceil(32)))*16;let column=(block%comptime!(n.div_ceil(32)))*32;
    let mut a_base=0usize;let mut b_base=0usize;let mut o_base=0usize;
    #[unroll] for reverse in 0usize..batch_shape.len() {
        let d=comptime!(batch_shape.len()-1-reverse);let coordinate=batch%comptime!(batch_shape[d]);
        batch/=comptime!(batch_shape[d]);
        a_base+=coordinate*comptime!(batch_left[d]);b_base+=coordinate*comptime!(batch_right[d]);
        o_base+=coordinate*comptime!(batch_out[d]);
    }
    let lane=UNIT_POS as usize;let local_row=lane/32;let local_column=lane%32;
    let mut a_tile=SharedMemory::<f32>::new(16usize*17usize);
    let mut b_tile=SharedMemory::<f32>::new(16usize*33usize);
    let mut sums=Array::<f32>::new(4usize);
    #[unroll] for slot in 0usize..4 {sums[slot]=0.0f32;}
    let mut reduction=0usize;
    while reduction<k {
        #[unroll] for part in 0usize..2 {
            let p=lane+part*128;let mut r=p/16;let mut j=p%16;
            if comptime!(am<ak) {r=p%16;j=p/16;}
            let mut value=0.0f32;
            if row+r<m && reduction+j<k {value=f32::cast_from(left[a_base+(row+r)*am+(reduction+j)*ak]);}
            a_tile[j*17+r]=value;
        }
        #[unroll] for part in 0usize..4 {
            let p=lane+part*128;let mut c=p%32;let mut j=p/32;
            if comptime!(bk<bn) {c=p/16;j=p%16;}
            let mut value=0.0f32;
            if column+c<n && reduction+j<k {value=f32::cast_from(right[b_base+(column+c)*bn+(reduction+j)*bk]);}
            b_tile[j*33+c]=value;
        }
        sync_ruda();
        #[unroll] for j in 0usize..16 {
            if comptime!(k%16==0) {
                let b=b_tile[j*33+local_column];
                #[unroll] for slot in 0usize..4 {
                    let a=a_tile[j*17+local_row+slot*4];
                    sums[slot]+=f32::cast_from(F::cast_from(a*b));
                }
            } else if reduction+j<k {
                let b=b_tile[j*33+local_column];
                #[unroll] for slot in 0usize..4 {
                    let a=a_tile[j*17+local_row+slot*4];
                    sums[slot]+=f32::cast_from(F::cast_from(a*b));
                }
            }
        }
        sync_ruda();reduction+=16usize;
    }
    #[unroll] for slot in 0usize..4 {
        let r=row+local_row+slot*4;
        if r<m && column+local_column<n {
            let mut value=sums[slot];if comptime!(mean) {value/=k as f32;}
            out[o_base+r*om+(column+local_column)*on]=F::cast_from(value);
        }
    }
}

fn prepare_contraction(client:&ComputeClient<CudaRuntime>,left:&View,right:&View,out:&View,
    shape:&[usize],mask:u32,mean:bool)->Option<PreparedKernel<CudaRuntime>>
{
    if mask.count_ones()!=1 {return None;}
    let rank=shape.len();let reduction=mask.trailing_zeros() as usize;
    let strides=|v:&View|->Vec<usize> {(0..rank).map(|d| {
        if d+v.shape.len()<rank {0}else {let axis=d+v.shape.len()-rank;if v.shape[axis]==1 {0}else {v.strides[axis]}}
    }).collect()};
    let a=strides(left);let b=strides(right);
    let rows=(0..rank).filter(|&d|d!=reduction && shape[d]>1 && a[d]!=0 && b[d]==0).collect::<Vec<_>>();
    let columns=(0..rank).filter(|&d|d!=reduction && shape[d]>1 && a[d]==0 && b[d]!=0).collect::<Vec<_>>();
    if rows.len()!=1 || columns.len()!=1 || a[reduction]==0 || b[reduction]==0 {return None;}
    let (row,column)=(rows[0],columns[0]);let (m,n,k)=(shape[row],shape[column],shape[reduction]);
    if m<8 || n<16 || k<16 {return None;}
    let mut output_strides=vec![0usize;rank];let mut size=1usize;
    for d in (0..rank).rev() {if d!=reduction {output_strides[d]=size;size*=shape[d];}}
    let batch_axes=(0..rank).filter(|&d|d!=row && d!=column && d!=reduction).collect::<Vec<_>>();
    let batch_shape=batch_axes.iter().map(|&d|shape[d]).collect::<Vec<_>>();
    let batch_left=batch_axes.iter().map(|&d|a[d]).collect::<Vec<_>>();
    let batch_right=batch_axes.iter().map(|&d|b[d]).collect::<Vec<_>>();
    let batch_out=batch_axes.iter().map(|&d|output_strides[d]).collect::<Vec<_>>();
    let blocks=m.div_ceil(16)*n.div_ceil(32)*batch_shape.iter().product::<usize>();
    let hardware=&client.properties().hardware;
    if hardware.max_units_per_ruda<128 || hardware.max_shared_memory_size<3200 || blocks>hardware.max_ruda_count.0 as usize {return None;}
    let grid=RudaCount::Static(u32::try_from(blocks).ok()?,1,1);
    macro_rules! run {($f:ty)=>{unsafe {contraction::prepare::<$f,CudaRuntime>(client,grid,RudaDim::new_1d(128),
        left.array_arg(),right.array_arg(),out.array_arg(),m,n,k,batch_shape,batch_left,batch_right,batch_out,
        a[row],a[reduction],b[column],b[reduction],output_strides[row],output_strides[column],mean)}};}
    Some(match left.dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),_=>unreachable!()})
}

pub(super) fn prepare(client:&ComputeClient<CudaRuntime>,left:&View,right:&View,out:&View,mask:u32,mean:bool)
    ->PreparedKernel<CudaRuntime>
{
    let rank=left.shape.len().max(right.shape.len());
    let shape:Vec<usize>=(0..rank).map(|d|{
        let a=if d+left.shape.len()<rank {1}else {left.shape[d+left.shape.len()-rank]};
        let b=if d+right.shape.len()<rank {1}else {right.shape[d+right.shape.len()-rank]};a.max(b)
    }).collect();
    if let Some(kernel)=prepare_contraction(client,left,right,out,&shape,mask,mean) {return kernel;}
    let width:usize=shape.iter().enumerate().filter(|(d,_)|mask&(1<<d)!=0).map(|(_,size)|size).product();
    let axes=super::graph_reduce::contiguous_axes(&shape,mask);
    let threads=if width<=128 || width<=1024 && out.len>=128 {32}else if width<=4096 {128}else {256};
    let vector=if shape[rank-1]%4==0 && [left,right].iter().all(|v|v.handle.offset_start.unwrap_or(0)%(4*element_bytes(v.dtype)) as u64==0) {4}else {1};
    let left_vector=left.shape.last().copied().unwrap_or(1)!=1;
    let right_vector=right.shape.last().copied().unwrap_or(1)!=1;
    let column_vector=if axes.is_some_and(|(_,inner)|inner>=128) && shape[rank-1]%4==0
        && [left,right].iter().all(|v|v.shape.last().copied().unwrap_or(1)==1
            || v.contiguous() && v.handle.offset_start.unwrap_or(0)%(4*element_bytes(v.dtype)) as u64==0)
        && out.contiguous() && out.handle.offset_start.unwrap_or(0)%(4*element_bytes(out.dtype)) as u64==0
        && client.properties().hardware.max_shared_memory_size>=4096 {4}else {1};
    let (path,blocks)=match axes {
        Some((_,1))=>(0,if threads==32 {out.len.div_ceil(4)}else {out.len}),
        Some((outer,inner)) if inner>=16=>(1,outer*inner.div_ceil(32*column_vector)),
        _=>(2,out.len),
    };
    let grid=RudaCount::Static(u32::try_from(blocks).expect("product reduction grid overflow"),1,1);
    let mut strides=vec![1usize;rank];for d in (0..rank-1).rev() {strides[d]=strides[d+1]*shape[d+1];}
    macro_rules! run {($f:ty)=>{unsafe {
        if path==0 {
            if vector==4 {suffix::prepare::<$f,CudaRuntime>(client,grid,RudaDim::new_1d(threads.max(128) as u32),4,
                left.arg(),right.arg(),out.array_arg(),shape,width,threads,left_vector,right_vector,mean)}
            else {suffix::prepare::<$f,CudaRuntime>(client,grid,RudaDim::new_1d(threads.max(128) as u32),1,
                left.arg(),right.arg(),out.array_arg(),shape,width,threads,left_vector,right_vector,mean)}
        } else if path==1 {
            if column_vector==4 {columns::prepare::<$f,CudaRuntime>(client,grid,RudaDim::new_2d(32,8),4,
                left.arg(),right.arg(),out.array_arg(),shape,width,axes.unwrap().1,mean,left_vector,right_vector)}
            else {columns::prepare::<$f,CudaRuntime>(client,grid,RudaDim::new_2d(32,8),1,
                left.arg(),right.arg(),out.array_arg(),shape,width,axes.unwrap().1,mean,left_vector,right_vector)}
        } else {
            arbitrary::prepare::<$f,CudaRuntime>(client,grid,RudaDim::new_1d(256),
                left.arg(),right.arg(),out.array_arg(),shape,strides,mask,width,mean)
        }
    }};}
    match left.dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),_=>unreachable!()}
}
