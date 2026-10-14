use super::*;
use ruda_core::{ir::{ElemType,FloatKind,features::MmaConfig},tensor::DType};
use ruda_kernel::tensor::allocation::empty_device_contiguous_dtype;

pub(super) fn supported(x:&RudaTensor<CudaRuntime>,experts:usize,columns:usize)->bool {
    let h=&x.client.properties().hardware;
    let cfg=MmaConfig {a_type:x.dtype.into(),b_type:x.dtype.into(),
        cd_type:ElemType::Float(FloatKind::F32).into(),m:16,n:16,k:16};
    matches!(x.dtype,DType::F16|DType::BF16) && h.plane_size_min==32 && h.plane_size_max==32
        && h.max_shared_memory_size>=2048 && x.client.features().matmul.cmma.contains(&cfg)
        && columns.max(x.meta.shape()[1]).div_ceil(16)<=h.max_ruda_count.0 as usize
        && x.meta.shape()[0].div_ceil(16)+experts<=h.max_ruda_count.1 as usize
        && columns.div_ceil(16)<=h.max_ruda_count.1 as usize
        && x.meta.shape()[0].div_ceil(128).max(1)<=h.max_ruda_count.0 as usize
        && experts<=h.max_ruda_count.1 as usize
}

pub(super) struct Rows {
    pub indices:RudaTensor<CudaRuntime>,
    pub offsets:RudaTensor<CudaRuntime>,
    pub tiles:RudaTensor<CudaRuntime>,
    pub plan:GroupedKernelPlan<CudaRuntime>,
}

// Stable row order is also the block-scaling order for the FP8 expert paths.
pub(super) fn prepare_rows(c:&ComputeClient<CudaRuntime>,ids:&RudaTensor<CudaRuntime>,start:u32,experts:usize)->Rows {
    let rows=ids.meta.num_elements();let chunks=rows.div_ceil(128).max(1);
    let allocate=|shape|empty_device_contiguous_dtype(c.clone(),ids.device.clone(),shape,DType::U32);
    let counts=allocate([experts,chunks].into());let indices=allocate([rows].into());
    let offsets=allocate([experts+1].into());let tiles=allocate([experts+1].into());
    let mut kernels=Vec::new();
    unsafe {
        kernels.push(count_rows::prepare::<CudaRuntime>(c,RudaCount::Static(chunks as u32,experts as u32,1),RudaDim::new_1d(32),
            ids.clone().into_array_arg(),counts.clone().into_array_arg(),start,chunks as u32,ids.dtype.into()));
        kernels.push(block_prefix::prepare::<CudaRuntime>(c,RudaCount::Static(experts as u32,1,1),RudaDim::new_1d(1),
            counts.clone().into_array_arg(),offsets.clone().into_array_arg(),chunks as u32));
        kernels.push(expert_prefix::prepare::<CudaRuntime>(c,RudaCount::Static(1,1,1),RudaDim::new_1d(1),
            offsets.clone().into_array_arg(),tiles.clone().into_array_arg()));
        if rows!=0 {
            kernels.push(map_rows::prepare::<CudaRuntime>(c,RudaCount::Static(chunks as u32,experts as u32,1),RudaDim::new_1d(32),
                ids.clone().into_array_arg(),counts.clone().into_array_arg(),offsets.clone().into_array_arg(),
                indices.clone().into_array_arg(),start,chunks as u32,ids.dtype.into()));
        }
    }
    let workspace=vec![counts,indices.clone(),offsets.clone(),tiles.clone()];
    Rows {indices,offsets,tiles,plan:GroupedKernelPlan {kernels,workspace}}
}

pub(super) fn prepare(c:&ComputeClient<CudaRuntime>,ts:&[RudaTensor<CudaRuntime>],backward:bool,ints:&[i64])
    ->GroupedKernelPlan<CudaRuntime> {
    let count=ints[1] as usize;let (inner,columns)=(ts[0].meta.shape()[1],ts[2].meta.shape()[0]);
    let mut rows=prepare_rows(c,&ts[1],ints[0] as u32,count);
    let input=&ts[if backward {count+2}else{0}];let output=&ts[if backward {count+3}else{count+2}];
    let width=if backward {inner}else{columns};
    unsafe {
        if output.meta.num_elements()!=0 {
            rows.plan.kernels.push(zero::prepare::<CudaRuntime>(c,
                RudaCount::Static(output.meta.num_elements().div_ceil(128) as u32,1,1),RudaDim::new_1d(128),
                output.clone().into_array_arg(),output.dtype.into()));
            rows.plan.kernels.push(project::prepare::<CudaRuntime>(c,
                RudaCount::Static(width.div_ceil(16) as u32,(input.meta.shape()[0].div_ceil(16)+count) as u32,1),RudaDim::new_1d(32),
                input.clone().into_array_arg(),ts[2..2+count].iter().map(|w|w.clone().into_array_arg()).collect(),
                rows.indices.clone().into_array_arg(),rows.offsets.clone().into_array_arg(),rows.tiles.clone().into_array_arg(),
                output.clone().into_array_arg(),inner as u32,columns as u32,backward,input.dtype.into()));
        }
        if backward {
            for expert in 0..count {
                let out=&ts[count+4+expert];if out.meta.num_elements()==0 {continue;}
                rows.plan.kernels.push(weight_gradient::prepare::<CudaRuntime>(c,
                    RudaCount::Static(inner.div_ceil(16) as u32,columns.div_ceil(16) as u32,1),RudaDim::new_1d(32),
                    ts[0].clone().into_array_arg(),ts[count+2].clone().into_array_arg(),rows.indices.clone().into_array_arg(),
                    rows.offsets.clone().into_array_arg(),out.clone().into_array_arg(),inner as u32,columns as u32,
                    expert as u32,ts[0].dtype.into()));
            }
        }
    }
    rows.plan
}

#[ruda(launch)]
fn count_rows<I:Int>(ids:&Array<I>,counts:&mut Array<u32>,start:u32,chunks:u32,#[define(I)] _dtype:StorageType) {
    let chunk=RUDA_POS_X;let expert=RUDA_POS_Y;let mut count=0u32;
    #[unroll]
    for i in 0u32..4u32 {
        let row=chunk*128+i*32+UNIT_POS;let mut hit=0u32;
        if (row as usize)<ids.len() {hit=u32::cast_from(i64::cast_from(ids[row as usize])==i64::cast_from(start)+i64::cast_from(expert));}
        count+=plane_sum(hit);
    }
    if UNIT_POS==0 {counts[(expert*chunks+chunk) as usize]=count;}
}

#[ruda(launch)]
fn block_prefix(counts:&mut Array<u32>,offsets:&mut Array<u32>,chunks:u32) {
    let expert=RUDA_POS_X;let mut total=0u32;let mut block=0u32;
    while block<chunks {let n=counts[(expert*chunks+block) as usize];counts[(expert*chunks+block) as usize]=total;total+=n;block+=1;}
    offsets[(expert+1) as usize]=total;
}

#[ruda(launch)]
fn expert_prefix(offsets:&mut Array<u32>,tiles:&mut Array<u32>) {
    let mut expert=1usize;let mut total=0u32;let mut tile=0u32;offsets[0]=0;tiles[0]=0;
    while expert<offsets.len() {
        let count=offsets[expert];total+=count;tile+=count/16+u32::cast_from(count%16!=0);
        offsets[expert]=total;tiles[expert]=tile;expert+=1;
    }
}

#[ruda(launch)]
fn map_rows<I:Int>(ids:&Array<I>,counts:&Array<u32>,offsets:&Array<u32>,indices:&mut Array<u32>,
    start:u32,chunks:u32,#[define(I)] _dtype:StorageType) {
    let chunk=RUDA_POS_X;let expert=RUDA_POS_Y;let mut base=offsets[expert as usize]+counts[(expert*chunks+chunk) as usize];
    #[unroll]
    for i in 0u32..4u32 {
        let row=chunk*128+i*32+UNIT_POS;let mut hit=0u32;
        if (row as usize)<ids.len() {hit=u32::cast_from(i64::cast_from(ids[row as usize])==i64::cast_from(start)+i64::cast_from(expert));}
        let rank=plane_exclusive_sum(hit);let count=plane_sum(hit);
        if hit!=0 {indices[(base+rank) as usize]=row;}
        base+=count;
    }
}

#[ruda(launch)]
pub(super) fn zero<F:Float>(output:&mut Array<F>,#[define(F)] _dtype:StorageType) {
    if ABSOLUTE_POS<output.len() {output[ABSOLUTE_POS]=F::cast_from(0.0f32);}
}

#[ruda(launch)]
fn project<F:Float>(input:&Array<F>,weights:&Sequence<Array<F>>,indices:&Array<u32>,offsets:&Array<u32>,
    tiles:&Array<u32>,output:&mut Array<F>,inner:u32,columns:u32,#[comptime] transpose:bool,#[define(F)] _dtype:StorageType) {
    let tile=RUDA_POS_Y;let col=RUDA_POS_X as usize*16;let lane=UNIT_POS as usize;
    if tile>=tiles[tiles.len()-1] {terminate!();}
    let mut selected=0u32;let mut begin=0usize;let mut end=0usize;
    #[unroll]
    for expert in 0..weights.len() {
        if tile>=tiles[expert] && tile<tiles[expert+1] {
            selected=expert as u32;begin=(offsets[expert]+(tile-tiles[expert])*16) as usize;end=offsets[expert+1] as usize;
        }
    }
    let width=if comptime!(transpose) {inner as usize}else{columns as usize};
    let contraction=if comptime!(transpose) {columns as usize}else{inner as usize};
    let mut left=SharedMemory::<F>::new_aligned(256usize,32usize);
    let mut right=SharedMemory::<F>::new_aligned(256usize,32usize);
    let mut result=SharedMemory::<f32>::new_aligned(256usize,32usize);
    let acc=cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,16usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0);
    let mut base=0usize;
    while base<contraction {
        #[unroll]
        for i in 0usize..8usize {
            let t=lane+i*32;let row=begin+t/16;let column=col+t/16;let k=base+t%16;
            let mut a=F::cast_from(0.0f32);let mut b=F::cast_from(0.0f32);
            if row<end && k<contraction {a=input[indices[row] as usize*contraction+k];}
            if column<width && k<contraction {
                #[unroll]
                for expert in 0..weights.len() {
                    if selected==expert as u32 {
                        let weight=&weights[expert];
                        let p=if comptime!(transpose) {k*inner as usize+column}else{column*inner as usize+k};
                        b=weight[p];
                    }
                }
            }
            left[t]=a;right[t]=b;
        }
        sync_ruda();
        let a=cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::A,16usize,16usize,16usize,cmma::MatrixLayout::RowMajor,&left.to_slice(),16);
        let b=cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::B,16usize,16usize,16usize,cmma::MatrixLayout::ColMajor,&right.to_slice(),16);
        cmma::execute::<F,F,f32,f32>(&a,&b,&acc,&acc);sync_ruda();
        if contraction-base<=16 {base=contraction;}else {base+=16;}
    }
    cmma::store(&mut result.to_slice_mut(),&acc,16,cmma::MatrixLayout::RowMajor);sync_ruda();
    #[unroll]
    for i in 0usize..8usize {
        let t=lane+i*32;let row=begin+t/16;let column=col+t%16;
        if row<end && column<width {let target=indices[row] as usize*width+column;output[target]=F::cast_from(result[t]);}
    }
}

#[ruda(launch)]
fn weight_gradient<F:Float>(input:&Array<F>,gradient:&Array<F>,indices:&Array<u32>,offsets:&Array<u32>,
    output:&mut Array<F>,inner:u32,columns:u32,expert:u32,#[define(F)] _dtype:StorageType) {
    let col=RUDA_POS_X as usize*16;let row=RUDA_POS_Y as usize*16;let lane=UNIT_POS as usize;
    let mut left=SharedMemory::<F>::new_aligned(256usize,32usize);
    let mut right=SharedMemory::<F>::new_aligned(256usize,32usize);
    let mut result=SharedMemory::<f32>::new_aligned(256usize,32usize);
    let acc=cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,16usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0);
    let mut base=offsets[expert as usize] as usize;let end=offsets[expert as usize+1] as usize;
    while base<end {
        #[unroll]
        for i in 0usize..8usize {
            let t=lane+i*32;let output_row=row+t/16;let column=col+t/16;let routed=base+t%16;
            let mut a=F::cast_from(0.0f32);let mut b=F::cast_from(0.0f32);
            if routed<end {
                let source=indices[routed] as usize;
                if output_row<columns as usize {a=gradient[source*columns as usize+output_row];}
                if column<inner as usize {b=input[source*inner as usize+column];}
            }
            left[t]=a;right[t]=b;
        }
        sync_ruda();
        let a=cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::A,16usize,16usize,16usize,cmma::MatrixLayout::RowMajor,&left.to_slice(),16);
        let b=cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::B,16usize,16usize,16usize,cmma::MatrixLayout::ColMajor,&right.to_slice(),16);
        cmma::execute::<F,F,f32,f32>(&a,&b,&acc,&acc);sync_ruda();
        if end-base<=16 {base=end;}else {base+=16;}
    }
    cmma::store(&mut result.to_slice_mut(),&acc,16,cmma::MatrixLayout::RowMajor);sync_ruda();
    #[unroll]
    for i in 0usize..8usize {
        let t=lane+i*32;let output_row=row+t/16;let column=col+t%16;
        if output_row<columns as usize && column<inner as usize {output[output_row*inner as usize+column]=F::cast_from(result[t]);}
    }
}
