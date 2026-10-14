use super::*;
use rublas::tensor_grouped::GroupedKernelPlan;
use ruda_kernel::tensor::RudaTensor;
mod tensorcore;
pub(super) mod quantized;
pub(super) mod float8;
pub(super) mod float8_block;

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_indexed_expert_api_version()->u32 {1}

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_indexed_expert_training_api_version()->u32 {1}

pub(super) fn validate(vs:&[View],backward:bool,ints:&[i64]) {
    assert!(ints.len()==2 || backward && ints.len()==4);
    assert!(ints[0]>=0 && ints[0]<=u32::MAX as i64 && (1..=16).contains(&ints[1]));
    let count=ints[1] as usize;
    let (need_x,need_w)=if ints.len()==4 {
        assert!(ints[2]==0 || ints[2]==1);assert!(ints[3]>=0 && ints[3]<(1i64<<count));
        (ints[2]!=0,ints[3])
    }else {(true,(1i64<<count)-1)};
    assert!(ints[0].checked_add(ints[1]).is_some_and(|end|end<=u32::MAX as i64));
    assert_eq!(vs.len(),if backward {2*count+4}else{count+3});
    for v in vs {super::operation_graph::packed(v);}
    let (x,ids,first)=(&vs[0],&vs[1],&vs[2]);
    assert_eq!(x.shape.len(),2);assert_eq!(first.shape.len(),2);
    assert!(x.dtype<=2 && matches!(ids.dtype,4|5));
    assert_eq!(ids.shape,vec![x.shape[0]]);
    assert!(x.shape[1]>0 && first.shape[0]>0 && first.shape[1]==x.shape[1]);
    for w in &vs[2..2+count] {assert_eq!(w.dtype,x.dtype);assert_eq!(w.shape,first.shape);}
    let result=&vs[count+2];
    assert_eq!(result.dtype,x.dtype);assert_eq!(result.shape,vec![x.shape[0],first.shape[0]]);
    if backward {
        assert_eq!(vs[count+3].shape,if need_x {x.shape.clone()}else {vec![0]});assert_eq!(vs[count+3].dtype,x.dtype);
        for (index,w) in vs[count+4..].iter().enumerate() {
            assert_eq!(w.shape,if need_w&(1<<index)!=0 {first.shape.clone()}else {vec![0]});assert_eq!(w.dtype,x.dtype);
        }
    }
}

pub(super) fn prepare(c:&ComputeClient<CudaRuntime>,vs:&[View],backward:bool,ints:&[i64])
    ->GroupedKernelPlan<CudaRuntime> {
    validate(vs,backward,ints);
    let ts:Vec<RudaTensor<CudaRuntime>>=vs.iter().map(|v|{
        let mut t=primitives::tensor(v);t.client=c.clone();t
    }).collect();
    let count=ints[1] as usize;
    let (inner,columns)=(vs[0].shape[1],vs[2].shape[0]);
    if tensorcore::supported(&ts[0],count,columns) {
        return tensorcore::prepare(c,&ts,backward,ints);
    }
    let mut kernels=Vec::new();let mut workspace=Vec::new();
    let grid=|n:usize|RudaCount::Static(u32::try_from(n.div_ceil(128)).expect("indexed expert grid overflow"),1,1);
    let output=if backward {count+3}else{count+2};
    if vs[output].len!=0 {
        kernels.push(project::prepare::<CudaRuntime>(c,grid(vs[output].len),RudaDim::new_1d(128),
            ts[if backward {count+2}else{0}].clone().into_array_arg(),ts[1].clone().into_array_arg(),
            ts[2..2+count].iter().map(|t|t.clone().into_array_arg()).collect(),ts[output].clone().into_array_arg(),
            inner as u32,columns as u32,ints[0] as u32,backward,ts[0].dtype.into(),ts[1].dtype.into()));
    }
    if backward && vs[count+4..].iter().any(|value|value.len!=0) {
        let ranges=ruda_kernel::tensor::allocation::empty_device_contiguous_dtype(c.clone(),ts[0].device.clone(),
            ruda_core::tensor::Shape::new([count,2]),ruda_core::tensor::DType::U32);
        kernels.push(unsafe {expert_ranges::prepare::<CudaRuntime>(c,RudaCount::Static(count as u32,1,1),RudaDim::new_1d(32),
            ts[1].clone().into_array_arg(),ranges.clone().into_array_arg(),ints[0] as u32,ts[1].dtype.into())});
        for expert in 0..count {
            if vs[count+4+expert].len==0 {continue;}
            let out=&ts[count+4+expert];
            kernels.push(weight_gradient::prepare::<CudaRuntime>(c,grid(vs[count+4+expert].len),RudaDim::new_1d(128),
                ts[0].clone().into_array_arg(),ts[1].clone().into_array_arg(),ts[count+2].clone().into_array_arg(),
                ranges.clone().into_array_arg(),out.clone().into_array_arg(),inner as u32,columns as u32,
                ints[0] as u32+expert as u32,expert as u32,ts[0].dtype.into(),ts[1].dtype.into()));
        }
        workspace.push(ranges);
    }
    GroupedKernelPlan {kernels,workspace}
}

pub(super) fn validate_activity(vs:&[View],ints:&[i64]) {
    assert_eq!(vs.len(),2);assert_eq!(ints.len(),2);
    assert!(ints[0]>=0 && ints[1]>0 && ints[1]<=u32::MAX as i64 && ints[0]<=u32::MAX as i64-ints[1]);
    for value in vs {super::operation_graph::packed(value);}
    assert_eq!(vs[0].shape.len(),1);assert!(matches!(vs[0].dtype,4|5));
    assert_eq!(vs[1].shape,vec![ints[1] as usize]);assert_eq!(vs[1].dtype,5);
}

pub(super) fn prepare_activity(c:&ComputeClient<CudaRuntime>,vs:&[View],ints:&[i64])->PreparedKernel<CudaRuntime> {
    validate_activity(vs,ints);let ids=primitives::tensor(&vs[0]);let out=primitives::tensor(&vs[1]);
    unsafe {activity::prepare::<CudaRuntime>(c,RudaCount::Static(ints[1] as u32,1,1),RudaDim::new_1d(32),
        ids.clone().into_array_arg(),out.into_array_arg(),ints[0] as u32,ids.dtype.into())}
}

#[ruda(launch)]
fn activity<I:Int>(ids:&Array<I>,output:&mut Array<i32>,start:u32,#[define(I)] _id_dtype:StorageType) {
    let expert=RUDA_POS_X;let lane=UNIT_POS;let mut row=lane;let mut found=0i32;
    while row<ids.len() {
        if i64::cast_from(ids[row])==i64::cast_from(start)+i64::cast_from(expert) {found=1;}
        if ids.len()-row<=32 {break;}
        row+=32;
    }
    found=plane_max(found);
    if lane==0 {output[expert]=found;}
}

#[ruda(launch)]
fn expert_ranges<I:Int>(ids:&Array<I>,output:&mut Array<u32>,start:u32,#[define(I)] _id_dtype:StorageType) {
    let expert=RUDA_POS_X;let lane=UNIT_POS;let mut row=lane;
    let mut first=ids.len();let mut end=0u32;
    while row<ids.len() {
        if i64::cast_from(ids[row])==i64::cast_from(start)+i64::cast_from(expert) {
            first=u32::min(first,row);end=u32::max(end,row+1);
        }
        if ids.len()-row<=32 {break;}
        row+=32;
    }
    first=plane_min(first);end=plane_max(end);
    if lane==0 {output[expert*2]=first;output[expert*2+1]=end;}
}

#[ruda(launch)]
fn project<F:Float,I:Int>(input:&Array<F>,ids:&Array<I>,weights:&Sequence<Array<F>>,output:&mut Array<F>,
    inner:u32,columns:u32,expert_start:u32,#[comptime] transpose:bool,
    #[define(F)] _dtype:StorageType,#[define(I)] _id_dtype:StorageType) {
    let p=ABSOLUTE_POS;if p>=output.len() {terminate!();}
    let width=if comptime!(transpose) {inner}else{columns};
    let contraction=if comptime!(transpose) {columns}else{inner};
    let row=p/width as usize;let column=p%width as usize;
    let selected=i64::cast_from(ids[row]);let mut sum=0.0f32;
    #[unroll]
    for expert in 0..weights.len() {
        if selected==i64::cast_from(expert_start)+i64::cast_from(expert) {
            let weight=&weights[expert];
            let mut k=0usize;
            while k<contraction as usize {
                let index=if comptime!(transpose) {k*inner as usize+column}else{column*inner as usize+k};
                sum=fma(f32::cast_from(input[row*contraction as usize+k]),f32::cast_from(weight[index]),sum);
                k+=1;
            }
        }
    }
    output[p]=F::cast_from(sum);
}

#[ruda(launch)]
fn weight_gradient<F:Float,I:Int>(input:&Array<F>,ids:&Array<I>,gradient:&Array<F>,ranges:&Array<u32>,output:&mut Array<F>,
    inner:u32,columns:u32,expert:u32,local_expert:u32,#[define(F)] _dtype:StorageType,#[define(I)] _id_dtype:StorageType) {
    let p=ABSOLUTE_POS;if p>=output.len() {terminate!();}
    let column=p/inner as usize;let k=p%inner as usize;
    let mut row=ranges[local_expert*2] as usize;let end=ranges[local_expert*2+1] as usize;let mut sum=0.0f32;
    while row<end {
        if i64::cast_from(ids[row])==i64::cast_from(expert) {
            sum=fma(f32::cast_from(input[row*inner as usize+k]),
                f32::cast_from(gradient[row*columns as usize+column]),sum);
        }
        row+=1;
    }
    output[p]=F::cast_from(sum);
}
