use super::*;
use ruda_core::{ir::{ElemType,FloatKind,features::MmaConfig},tensor::DType};
use ruda_kernel::quantization::packed::{PackedTensor,PackedFormat,load_grouped};

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_indexed_quantized_expert_api_version()->u32 {1}

pub(crate) fn validate(vs:&[View],ints:&[i64]) {
    assert!(ints.len()>=3);let count=ints[1];assert!((1..=16).contains(&count));
    assert!(ints[0]>=0 && ints[0]<=u32::MAX as i64-count && matches!(ints[2],0|1));
    let count=count as usize;assert_eq!(ints.len(),3+6*count);assert_eq!(vs.len(),3+4*count);
    for v in vs {crate::operation_graph::packed(v);}
    let (x,ids,out)=(&vs[0],&vs[1],vs.last().unwrap());
    assert_eq!(x.shape.len(),2);assert!(x.dtype<=2 && matches!(ids.dtype,4|5));
    assert_eq!(ids.shape,vec![x.shape[0]]);assert_eq!(out.dtype,x.dtype);
    for e in 0..count {
        let p=&ints[3+6*e..9+6*e];let w=&vs[2+4*e..6+4*e];
        assert!(p[1..5].iter().all(|&n|n>=0 && n<=u32::MAX as i64));
        assert!(p[1]>0 && p[2]>0 && p[3]>0 && matches!(p[5],0|1));
        assert!(p[2].checked_mul(p[3]).is_some_and(|n|n<=u32::MAX as i64));
        assert_eq!(x.shape[1],p[if ints[2]==0 {2}else{3}] as usize);
        assert_eq!(out.shape,vec![x.shape[0],p[if ints[2]==0 {3}else{2}] as usize]);
        assert_eq!(w[0].dtype,8);assert_eq!(w[1].dtype,0);assert_eq!(w[2].dtype,0);assert_eq!(w[3].dtype,5);
        if p[0]==0 {
            assert!(p[1]%2==0 && p[4]==0 && w[3].len==0);
            let n=(p[2]*p[3]) as usize;assert_eq!(w[0].len,n.div_ceil(2));
            assert_eq!(w[1].len,n.div_ceil(p[1] as usize));assert_eq!(w[2].len,16);
        }else {
            PackedTensor::new(primitives::tensor(&w[0]),primitives::tensor(&w[1]),primitives::tensor(&w[2]),
                primitives::tensor(&w[3]),[p[3] as usize,p[2] as usize].into(),PackedFormat::from_code(p[0] as u32),p[1] as usize);
            assert_eq!(w[1].len,p[3] as usize*p[4] as usize);
        }
        if p[5]==1 {assert!(matches!(x.dtype,1|2));}
    }
}

pub(crate) fn prepare(c:&ComputeClient<CudaRuntime>,vs:&[View],ints:&[i64])->GroupedKernelPlan<CudaRuntime> {
    validate(vs,ints);let ts:Vec<_>=vs.iter().map(|v|{let mut t=primitives::tensor(v);t.client=c.clone();t}).collect();
    let (x,ids,out)=(&ts[0],&ts[1],ts.last().unwrap());let count=ints[1] as usize;
    if out.meta.num_elements()==0 {return GroupedKernelPlan {kernels:Vec::new(),workspace:Vec::new()};}
    let h=&c.properties().hardware;
    assert!(h.plane_size_min==32 && h.plane_size_max==32 && x.meta.shape()[0].div_ceil(128)<=h.max_ruda_count.0 as usize);
    let mut rows=tensorcore::prepare_rows(c,ids,ints[0] as u32,count);
    let size=out.meta.num_elements();let width=out.meta.shape()[1];
    unsafe {
        rows.plan.kernels.push(tensorcore::zero::prepare::<CudaRuntime>(c,
            RudaCount::Static(size.div_ceil(128) as u32,1,1),RudaDim::new_1d(128),out.clone().into_array_arg(),out.dtype.into()));
        for e in 0..count {
            let p=&ints[3+6*e..9+6*e];let w=&ts[2+4*e..6+4*e];
            let core=p[5]!=0;
            if core {
                let cfg=MmaConfig {a_type:x.dtype.into(),b_type:x.dtype.into(),cd_type:ElemType::Float(FloatKind::F32).into(),m:16,n:16,k:16};
                assert!(matches!(x.dtype,DType::F16|DType::BF16) && h.max_shared_memory_size>=2048
                    && c.features().matmul.cmma.contains(&cfg) && width.div_ceil(16)<=h.max_ruda_count.0 as usize,
                    "indexed quantized expert Tensor Core configuration unavailable");
                let blocks=x.meta.shape()[0].div_ceil(16).min(128).min(h.max_ruda_count.1 as usize);
                assert!(blocks>0);
                rows.plan.kernels.push(cooperative::prepare::<CudaRuntime>(c,
                    RudaCount::Static(width.div_ceil(16) as u32,blocks as u32,1),RudaDim::new_1d(32),
                    x.clone().into_array_arg(),w[0].clone().into_array_arg(),w[1].clone().into_array_arg(),w[2].clone().into_array_arg(),
                    w[3].clone().into_array_arg(),rows.indices.clone().into_array_arg(),rows.offsets.clone().into_array_arg(),
                    out.clone().into_array_arg(),e as u32,(blocks*16) as u32,p[2] as usize,p[3] as usize,p[1] as usize,
                    p[4] as usize,p[0] as u32,w[3].meta.num_elements()!=0,ints[2]!=0,x.dtype.into()));
            }else {
                let blocks=size.div_ceil(128).min(2048).min(h.max_ruda_count.0 as usize);assert!(blocks>0);
                rows.plan.kernels.push(scalar::prepare::<CudaRuntime>(c,RudaCount::Static(blocks as u32,1,1),RudaDim::new_1d(128),
                    x.clone().into_array_arg(),w[0].clone().into_array_arg(),w[1].clone().into_array_arg(),w[2].clone().into_array_arg(),
                    w[3].clone().into_array_arg(),rows.indices.clone().into_array_arg(),rows.offsets.clone().into_array_arg(),
                    out.clone().into_array_arg(),e as u32,(blocks*128) as u32,p[2] as usize,p[3] as usize,p[1] as usize,
                    p[4] as usize,p[0] as u32,w[3].meta.num_elements()!=0,ints[2]!=0,x.dtype.into()));
            }
        }
    }
    rows.plan
}

#[ruda]
fn weight(codes:&Array<u8>,scales:&Array<f32>,offsets:&Array<f32>,groups:&Array<i32>,row:usize,col:usize,
    #[comptime] width:usize,#[comptime] group:usize,#[comptime] parameters:usize,
    #[comptime] format:u32,#[comptime] indexed:bool)->f32 {
    if comptime!(format==0) {
        let p=row*width+col;let byte=codes[p/2] as u32;
        let code=if p%2==0 {byte>>4}else{byte&15};
        offsets[code as usize]*scales[p/group]
    }else {load_grouped(codes,scales,offsets,groups,row,col,width,group,parameters,format,indexed)}
}

#[ruda(launch)]
fn scalar<F:Float>(x:&Array<F>,codes:&Array<u8>,scales:&Array<f32>,offsets:&Array<f32>,groups:&Array<i32>,
    indices:&Array<u32>,segments:&Array<u32>,out:&mut Array<F>,expert:u32,step:u32,
    #[comptime] inputs:usize,#[comptime] outputs:usize,#[comptime] group:usize,#[comptime] parameters:usize,
    #[comptime] format:u32,#[comptime] indexed:bool,#[comptime] transpose:bool,#[define(F)] _dtype:StorageType) {
    let width=comptime!(if transpose {inputs}else{outputs});let inner=comptime!(if transpose {outputs}else{inputs});
    let begin=segments[expert as usize] as usize;let end=segments[expert as usize+1] as usize;let mut p=ABSOLUTE_POS as usize;
    while p<(end-begin)*width {
        let row=indices[begin+p/width] as usize;let col=p%width;let mut sum=0.0f32;let mut k=0usize;
        while k<inner {
            let wr=if comptime!(transpose) {k}else{col};let wc=if comptime!(transpose) {col}else{k};
            let w=weight(codes,scales,offsets,groups,wr,wc,inputs,group,parameters,format,indexed);
            sum=fma(f32::cast_from(x[row*inner+k]),w,sum);k+=1;
        }
        out[row*width+col]=F::cast_from(sum);
        if (end-begin)*width-p<=step as usize {p=(end-begin)*width;}else {p+=step as usize;}
    }
}

#[ruda(launch)]
fn cooperative<F:Float>(x:&Array<F>,codes:&Array<u8>,scales:&Array<f32>,offsets:&Array<f32>,groups:&Array<i32>,
    indices:&Array<u32>,segments:&Array<u32>,out:&mut Array<F>,expert:u32,step:u32,
    #[comptime] inputs:usize,#[comptime] outputs:usize,#[comptime] group:usize,#[comptime] parameters:usize,
    #[comptime] format:u32,#[comptime] indexed:bool,#[comptime] transpose:bool,#[define(F)] _dtype:StorageType) {
    let width=comptime!(if transpose {inputs}else{outputs});let inner=comptime!(if transpose {outputs}else{inputs});
    let col=RUDA_POS_X as usize*16;let lane=UNIT_POS as usize;
    let mut row_base=segments[expert as usize] as usize+RUDA_POS_Y as usize*16;let end=segments[expert as usize+1] as usize;
    let mut left=SharedMemory::<F>::new_aligned(256usize,32usize);let mut right=SharedMemory::<F>::new_aligned(256usize,32usize);
    let mut result=SharedMemory::<f32>::new_aligned(256usize,32usize);
    while row_base<end {
        let acc=cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,16usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0);
        let mut base=0usize;
        while base<inner {
            #[unroll]
            for i in 0usize..8usize {
                let t=lane+i*32;let row=row_base+t/16;let column=col+t/16;let k=base+t%16;
                let mut a=F::cast_from(0.0f32);let mut b=F::cast_from(0.0f32);
                if row<end && k<inner {a=x[indices[row] as usize*inner+k];}
                if column<width && k<inner {
                    let wr=if comptime!(transpose) {k}else{column};let wc=if comptime!(transpose) {column}else{k};
                    b=F::cast_from(weight(codes,scales,offsets,groups,wr,wc,inputs,group,parameters,format,indexed));
                }
                left[t]=a;right[t]=b;
            }
            sync_ruda();
            let a=cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::A,16usize,16usize,16usize,cmma::MatrixLayout::RowMajor,&left.to_slice(),16);
            let b=cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::B,16usize,16usize,16usize,cmma::MatrixLayout::ColMajor,&right.to_slice(),16);
            cmma::execute::<F,F,f32,f32>(&a,&b,&acc,&acc);sync_ruda();
            if inner-base<=16 {base=inner;}else {base+=16;}
        }
        cmma::store(&mut result.to_slice_mut(),&acc,16,cmma::MatrixLayout::RowMajor);sync_ruda();
        #[unroll]
        for i in 0usize..8usize {
            let t=lane+i*32;let row=row_base+t/16;let column=col+t%16;
            if row<end && column<width {out[indices[row] as usize*width+column]=F::cast_from(result[t]);}
        }
        sync_ruda();if end-row_base<=step as usize {row_base=end;}else {row_base+=step as usize;}
    }
}
