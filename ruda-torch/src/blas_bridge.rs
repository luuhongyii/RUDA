use super::*;
use super::scientific::{real_tensor,complex_parts,copy_real,copy_complex};
use ruda_core::tensor::{DType,Shape};
use ruda_kernel::tensor::{RudaTensor,allocation::empty_device_contiguous_dtype};
use ruda_kernel::dsl::{self as kernel_dsl,calculate_ruda_count_elemwise};
use rublas::{tensor_matmul::{self,ComplexGemmOptions,TriangularSolveOptions,GemmOptions,MatmulStrategy,
    F32MathMode,GemmActivation,FusedGemmStrategy},tensor_grouped::{GroupedStrategy,grouped_matmul_nt_with_strategy}};

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_blas_api_version()->u32 {1}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_blas(op:u32,ds:*const Descriptor,count:usize,
    ints:*const i64,ni:usize,scalars:*const f64,ns:usize)->i32 {
    checked(|| {
        assert!(!ds.is_null() && (ni==0 || !ints.is_null()) && (ns==0 || !scalars.is_null()));
        let xs=unsafe {crate::abi::slice(ds,count)}.iter().map(|d|unsafe {crate::abi::view(d)}).collect::<Vec<_>>();
        let ii=if ni==0 {&[][..]}else {unsafe {crate::abi::slice(ints,ni)}};
        let ss=if ns==0 {&[][..]}else {unsafe {crate::abi::slice(scalars,ns)}};
        for x in &xs {
            assert!(x.len<=u32::MAX as usize && x.shape.iter().all(|&d|d<=u32::MAX as usize),"BLAS U32 indexing limit exceeded");
            if x.len!=0 {assert!(x.shape.iter().zip(&x.strides).try_fold(1usize,|span,(&d,&s)|
                (d-1).checked_mul(s).and_then(|n|span.checked_add(n))).is_some_and(|span|span<=u32::MAX as usize),
                "BLAS strided span exceeds U32 indexing");}
        }
        match op {
            0=>{
                assert!(xs.len()==3 || xs.len()==4);assert_eq!(ii.len(),6);assert_eq!(ss.len(),4);
                let a=&xs[0];let b=&xs[1];let out=xs.last().unwrap();let has_c=xs.len()==4;
                assert_eq!(a.dtype,b.dtype);assert_eq!(a.dtype,out.dtype);
                if a.dtype==10 || a.dtype==11 {
                    math(ii[4]);assert!(ii[5]==0 || ii[5]==1,"complex GEMM has no Tensor Core strategy; use auto/scalar");
                    let (ar,ai)=complex_parts(a);let (br,bi)=complex_parts(b);
                    let c=has_c.then(||complex_parts(&xs[2]));
                    let (re,im)=tensor_matmul::complex_gemm(ar,ai,br,bi,c,ComplexGemmOptions {
                        alpha:[ss[0],ss[1]],beta:[ss[2],ss[3]],transpose_lhs:flag(ii[0]),transpose_rhs:flag(ii[1]),
                        conjugate_lhs:flag(ii[2]),conjugate_rhs:flag(ii[3])}).expect("complex GEMM setup failed");
                    copy_complex(&re,&im,out);
                } else {
                    assert!(a.dtype==0 || a.dtype==9);assert_eq!(ss[1],0.);assert_eq!(ss[3],0.);
                    flag(ii[2]);flag(ii[3]);
                    let result=tensor_matmul::gemm(real_tensor(a),real_tensor(b),has_c.then(||real_tensor(&xs[2])),
                        GemmOptions {alpha:ss[0],beta:ss[2],transpose_lhs:flag(ii[0]),transpose_rhs:flag(ii[1]),
                            f32_math:math(ii[4])},matmul_strategy(ii[5])).expect("real GEMM setup failed");
                    copy_real(&result,out);
                }
            }
            1=>{
                assert_eq!(xs.len(),4);assert_eq!(ii.len(),2);assert_eq!(ss.len(),0);
                let activation=match ii[0] {0=>GemmActivation::None,1=>GemmActivation::Relu,2=>GemmActivation::Gelu,_=>panic!("invalid GEMM activation")};
                let strategy=match ii[1] {0=>FusedGemmStrategy::Auto,1=>FusedGemmStrategy::Scalar,2=>FusedGemmStrategy::TensorCore,_=>panic!("invalid fused GEMM strategy")};
                let out_dtype=real_tensor(&xs[3]).dtype;
                let result=tensor_matmul::matmul_bias_activation(real_tensor(&xs[0]),real_tensor(&xs[1]),real_tensor(&xs[2]),
                    activation,strategy,out_dtype).expect("fused GEMM setup failed");
                copy_real(&result,&xs[3]);
            }
            2=>{
                assert_eq!(xs.len(),4);assert_eq!(ii.len(),1);assert_eq!(ss.len(),0);
                assert_eq!(xs[0].shape.len(),2);assert_eq!(xs[1].shape.len(),3);assert_eq!(xs[2].shape,vec![xs[0].shape[0]]);
                let ids=row_ids(&xs[2],xs[1].shape[0]);
                let result=grouped_matmul_nt_with_strategy(real_tensor(&xs[0]),real_tensor(&xs[1]),ids,
                    grouped_strategy(ii[0])).expect("grouped GEMM setup failed");copy_real(&result,&xs[3]);
            }
            3=>{
                assert_eq!(xs.len(),6);assert_eq!(ii.len(),0);assert_eq!(ss.len(),0);
                let (x,w,ids,g,dx,dw)=(&xs[0],&xs[1],&xs[2],&xs[3],&xs[4],&xs[5]);
                assert!(x.shape.len()==2 && w.shape.len()==3 && g.shape==vec![x.shape[0],w.shape[1]]);
                assert_eq!(w.shape[2],x.shape[1]);assert_eq!(ids.shape,vec![x.shape[0]]);assert_eq!(dx.shape,x.shape);assert_eq!(dw.shape,w.shape);
                assert_eq!(x.dtype,w.dtype);assert_eq!(x.dtype,g.dtype);assert_eq!(dx.dtype,x.dtype);assert_eq!(dw.dtype,w.dtype);
                assert!(matches!(x.dtype,0|1|2),"grouped backward requires FP32/FP16/BF16 storage with FP32 accumulation");
                let ids=row_ids(ids,w.shape[0]);let xt=real_tensor(x);let wt=real_tensor(w);let gt=real_tensor(g);
                let dxt=real_tensor(dx);let dwt=real_tensor(dw);let c=client();
                if dx.len!=0 {let dim=RudaDim::new(c.properties(),dx.len);
                    arbitrary_dinput::launch::<CudaRuntime>(&c,calculate_ruda_count_elemwise(&c,dx.len,dim),dim,
                        wt.clone().into_tensor_arg(),gt.clone().into_tensor_arg(),ids.clone().into_array_arg(),dxt.into_tensor_arg(),
                        w.shape[0] as u32,w.shape[1] as u32,x.shape[1] as u32,xt.dtype.into());}
                if dw.len!=0 {let dim=RudaDim::new(c.properties(),dw.len);
                    arbitrary_dweight::launch::<CudaRuntime>(&c,calculate_ruda_count_elemwise(&c,dw.len,dim),dim,
                        xt.clone().into_tensor_arg(),gt.into_tensor_arg(),ids.into_array_arg(),dwt.into_tensor_arg(),
                        x.shape[0] as u32,w.shape[1] as u32,x.shape[1] as u32,xt.dtype.into());}
            }
            4=>{
                assert_eq!(xs.len(),3);assert_eq!(ii.len(),4);assert_eq!(ss.len(),1);
                let result=tensor_matmul::triangular_solve(real_tensor(&xs[0]),real_tensor(&xs[1]),TriangularSolveOptions {
                    left:flag(ii[0]),upper:flag(ii[1]),transpose:flag(ii[2]),unit_diagonal:flag(ii[3]),alpha:ss[0]})
                    .expect("triangular solve setup failed");copy_real(&result,&xs[2]);
            }
            5=>{
                assert_eq!(xs.len(),2);assert_eq!(ii.len(),1);assert_eq!(ss.len(),0);
                if xs[0].dtype==10 || xs[0].dtype==11 {
                    let (re,im)=complex_parts(&xs[0]);let rr=reduce_part(&re,&xs[1].shape,false);
                    let ri=reduce_part(&im,&xs[1].shape,flag(ii[0]));copy_complex(&rr,&ri,&xs[1]);
                } else {let result=reduce_part(&real_tensor(&xs[0]),&xs[1].shape,false);copy_real(&result,&xs[1]);}
            }
            6=>{
                assert_eq!(xs.len(),2);assert_eq!(ii.len(),0);assert_eq!(ss.len(),2);
                assert_eq!(xs[0].dtype,xs[1].dtype);assert_eq!(xs[0].shape,xs[1].shape);
                let complex=xs[0].dtype==10 || xs[0].dtype==11;
                let (re,im)=if complex {complex_parts(&xs[0])}else {let re=real_tensor(&xs[0]);assert_eq!(ss[1],0.);(re.clone(),re)};
                let rr=empty_device_contiguous_dtype(re.client.clone(),re.device.clone(),re.meta.shape().clone(),re.dtype);
                let ri=if complex {empty_device_contiguous_dtype(re.client.clone(),re.device.clone(),re.meta.shape().clone(),re.dtype)}else {rr.clone()};
                if xs[0].len!=0 {let dim=RudaDim::new(re.client.properties(),xs[0].len);let grid=calculate_ruda_count_elemwise(&re.client,xs[0].len,dim);
                    match re.dtype {
                        DType::F32=>scale_kernel::launch::<f32,CudaRuntime>(&re.client,grid,dim,re.clone().into_tensor_arg(),im.into_tensor_arg(),rr.clone().into_array_arg(),ri.clone().into_array_arg(),ss[0] as f32,ss[1] as f32,complex),
                        DType::F64=>scale_kernel::launch::<f64,CudaRuntime>(&re.client,grid,dim,re.clone().into_tensor_arg(),im.into_tensor_arg(),rr.clone().into_array_arg(),ri.clone().into_array_arg(),ss[0],ss[1],complex),_=>panic!("BLAS scale requires F32/F64 components"),
                    }
                }
                if complex {copy_complex(&rr,&ri,&xs[1]);}else {copy_real(&rr,&xs[1]);}
            }
            7|8|9=>{
                assert!(ss.is_empty());let c=client().fixed_execution_queue();
                let plan=indexed_experts::prepare(&c,&xs,op!=7,ii);
                for node in plan.kernels {let (task,count,arguments,client)=node.into_parts();client.launch(task,count,arguments);}
            }
            14|15=>{
                assert!(ss.is_empty());let c=client().fixed_execution_queue();
                let plan=indexed_experts::float8_block::prepare(&c,&xs,ii,op==15);
                for node in plan.kernels {let (task,count,arguments,client)=node.into_parts();client.launch(task,count,arguments);}
            }
            12|13=>{
                assert!(ss.is_empty());let c=client().fixed_execution_queue();
                let plan=if op==12 {indexed_experts::float8::prepare_statistics(&c,&xs,ii)}
                    else {indexed_experts::float8::prepare_product(&c,&xs,ii)};
                for node in plan.kernels {let (task,count,arguments,client)=node.into_parts();client.launch(task,count,arguments);}
            }
            11=>{
                assert!(ss.is_empty());let c=client().fixed_execution_queue();
                let plan=indexed_experts::quantized::prepare(&c,&xs,ii);
                for node in plan.kernels {let (task,count,arguments,client)=node.into_parts();client.launch(task,count,arguments);}
            }
            10=>{
                assert!(ss.is_empty());let c=client().fixed_execution_queue();
                let node=indexed_experts::prepare_activity(&c,&xs,ii);
                let (task,count,arguments,client)=node.into_parts();client.launch(task,count,arguments);
            }
            _=>panic!("unknown BLAS command"),
        }
        RUBLAS_CALLS.fetch_add(1,Ordering::Relaxed);finish_dispatch(&client());
    })
}

fn flag(i:i64)->bool {assert!(i==0 || i==1);i!=0}
fn math(i:i64)->F32MathMode {match i {0=>F32MathMode::Strict,1=>F32MathMode::AllowTf32,_=>panic!("invalid F32 math policy")}}
fn matmul_strategy(i:i64)->MatmulStrategy {match i {0=>MatmulStrategy::Ruda,1=>MatmulStrategy::Naive,
    2=>MatmulStrategy::CmmaResidueFirst,_=>panic!("invalid matmul strategy")}}
fn grouped_strategy(i:i64)->GroupedStrategy {match i {0=>GroupedStrategy::Auto,1=>GroupedStrategy::Scalar,
    2=>GroupedStrategy::TensorCore,_=>panic!("invalid grouped strategy")}}

pub(super) fn prepare_grouped(c:&ComputeClient<CudaRuntime>,views:&[View],backward:bool,integers:&[i64])
    ->rublas::tensor_grouped::GroupedKernelPlan<CudaRuntime>
{
    use rublas::tensor_grouped::{GroupedKernelPlan,prepare_grouped_matmul_nt};
    let ts:Vec<_>=views.iter().map(|v|{let mut t=primitives::tensor(v);t.client=c.clone();t}).collect();
    let (m,k,e,n)=(views[0].shape[0],views[0].shape[1],views[1].shape[0],views[1].shape[1]);
    let ids=empty_device_contiguous_dtype(c.clone(),ts[0].device.clone(),[m].into(),DType::U32);
    let mut kernels=Vec::new();
    if m!=0 {
        let dim=RudaDim::new(c.properties(),m);
        kernels.push(convert_ids::prepare::<CudaRuntime>(c,calculate_ruda_count_elemwise(c,m,dim),dim,
            ts[2].clone().into_tensor_arg(),ids.clone().into_array_arg(),e as u32,ts[2].dtype.into()));
    }
    let mut workspace=vec![ids.clone()];
    if backward {
        if views[4].len!=0 {
            let dim=RudaDim::new(c.properties(),views[4].len);
            kernels.push(arbitrary_dinput::prepare::<CudaRuntime>(c,calculate_ruda_count_elemwise(c,views[4].len,dim),dim,
                ts[1].clone().into_tensor_arg(),ts[3].clone().into_tensor_arg(),ids.clone().into_array_arg(),ts[4].clone().into_tensor_arg(),
                e as u32,n as u32,k as u32,ts[0].dtype.into()));
        }
        if views[5].len!=0 {
            let dim=RudaDim::new(c.properties(),views[5].len);
            kernels.push(arbitrary_dweight::prepare::<CudaRuntime>(c,calculate_ruda_count_elemwise(c,views[5].len,dim),dim,
                ts[0].clone().into_tensor_arg(),ts[3].clone().into_tensor_arg(),ids.into_array_arg(),ts[5].clone().into_tensor_arg(),
                m as u32,n as u32,k as u32,ts[0].dtype.into()));
        }
    } else {
        let prepared=unsafe {prepare_grouped_matmul_nt(&ts[0],&ts[1],&ids,&ts[3],grouped_strategy(integers[0]))}
            .expect("grouped graph preparation failed");
        kernels.extend(prepared.kernels);workspace.extend(prepared.workspace);
    }
    GroupedKernelPlan {kernels,workspace}
}

fn row_ids(ids:&View,experts:usize)->RudaTensor<CudaRuntime> {
    assert!(ids.dtype==4 || ids.dtype==5);assert_eq!(ids.shape.len(),1);
    let t=real_tensor(ids);let out=empty_device_contiguous_dtype(t.client.clone(),t.device.clone(),Shape::from(ids.shape.clone()),DType::U32);
    if ids.len!=0 {let dim=RudaDim::new(t.client.properties(),ids.len);
        convert_ids::launch::<CudaRuntime>(&t.client,calculate_ruda_count_elemwise(&t.client,ids.len,dim),dim,
            t.clone().into_tensor_arg(),out.clone().into_array_arg(),experts as u32,t.dtype.into());}
    out
}

#[ruda(launch)]
fn convert_ids<I:Int>(input:&Tensor<I>,out:&mut Array<u32>,experts:u32,#[define(I)] _dtype:StorageType) {
    let p=ABSOLUTE_POS;if p>=out.len() {terminate!();}
    let id=i64::cast_from(input[p*input.stride(0)]);let mut value=4294967295u32;
    if id>=0i64 && id<i64::cast_from(experts) {value=u32::cast_from(id);}out[p]=value;
}

#[ruda(launch)]
fn arbitrary_dinput<F:Float>(weights:&Tensor<F>,grad:&Tensor<F>,ids:&Array<u32>,out:&mut Tensor<F>,
    experts:u32,columns:u32,inner:u32,#[define(F)] _dtype:StorageType) {
    let p=ABSOLUTE_POS;if p>=out.len() {terminate!();}let row=p/inner as usize;let kk=p%inner as usize;
    let expert=ids[row];let mut sum=0.0f32;
    if expert<experts {let mut j=0usize;while j<columns as usize {
        sum=fma(f32::cast_from(grad[row*grad.stride(0)+j*grad.stride(1)]),
            f32::cast_from(weights[expert as usize*weights.stride(0)+j*weights.stride(1)+kk*weights.stride(2)]),sum);j+=1;
    }}out[row*out.stride(0)+kk*out.stride(1)]=F::cast_from(sum);
}

#[ruda(launch)]
fn arbitrary_dweight<F:Float>(input:&Tensor<F>,grad:&Tensor<F>,ids:&Array<u32>,out:&mut Tensor<F>,
    rows:u32,columns:u32,inner:u32,#[define(F)] _dtype:StorageType) {
    let p=ABSOLUTE_POS;if p>=out.len() {terminate!();}let kk=p%inner as usize;let col=p/inner as usize%columns as usize;
    let expert=p/(inner as usize*columns as usize);let mut sum=0.0f32;let mut row=0usize;
    while row<rows as usize {if ids[row] as usize==expert {
        sum=fma(f32::cast_from(input[row*input.stride(0)+kk*input.stride(1)]),
            f32::cast_from(grad[row*grad.stride(0)+col*grad.stride(1)]),sum);
    }row+=1;}out[expert*out.stride(0)+col*out.stride(1)+kk*out.stride(2)]=F::cast_from(sum);
}

fn reduce_part(input:&RudaTensor<CudaRuntime>,shape:&[usize],negate:bool)->RudaTensor<CudaRuntime> {
    assert!(shape.len()<=input.meta.num_dims());let rank=input.meta.num_dims();let lead=rank-shape.len();
    for (i,&d) in shape.iter().enumerate() {assert!(d==1 || d==input.meta.shape()[lead+i]);}
    let out=empty_device_contiguous_dtype(input.client.clone(),input.device.clone(),Shape::from(shape.to_vec()),input.dtype);
    let count=shape.iter().product::<usize>();if count==0 {return out;}
    let dim=RudaDim::new(input.client.properties(),count);let grid=calculate_ruda_count_elemwise(&input.client,count,dim);
    match input.dtype {DType::F32=>reduce_kernel::launch::<f32,CudaRuntime>(&input.client,grid,dim,input.clone().into_tensor_arg(),out.clone().into_tensor_arg(),negate),
        DType::F64=>reduce_kernel::launch::<f64,CudaRuntime>(&input.client,grid,dim,input.clone().into_tensor_arg(),out.clone().into_tensor_arg(),negate),
        _=>panic!("BLAS reduction requires F32/F64 components")};out
}

#[ruda(launch)]
fn reduce_kernel<F:Float+RudaElement>(input:&Tensor<F>,out:&mut Tensor<F>,#[comptime] negate:bool) {
    let p=ABSOLUTE_POS;if p>=out.len() {terminate!();}let lead=input.rank()-out.rank();
    let mut reductions=1usize;let mut fixed=0usize;let mut coordinate=p;let mut axis=input.rank();
    while axis>0 {axis-=1;
        if axis<lead {reductions*=input.shape(axis);}else {let oa=axis-lead;
            let q=coordinate%out.shape(oa);coordinate/=out.shape(oa);
            if out.shape(oa)==1 {reductions*=input.shape(axis);}else {fixed+=q*input.stride(axis);}
        }
    }
    let mut value=F::new(0.0f32);let mut j=0usize;
    while j<reductions {let mut rest=j;let mut offset=fixed;let mut a=input.rank();
        while a>0 {a-=1;let mut reduce=a<lead;if a>=lead {reduce=out.shape(a-lead)==1;}
            if reduce {offset+=(rest%input.shape(a))*input.stride(a);rest/=input.shape(a);}
        }value+=input[offset];j+=1;
    }if comptime!(negate) {value=-value;}out[p]=value;
}

#[ruda(launch)]
fn scale_kernel<F:Float+RudaElement>(re:&Tensor<F>,im:&Tensor<F>,rr:&mut Array<F>,ri:&mut Array<F>,ar:F,ai:F,#[comptime] complex:bool) {
    let p=ABSOLUTE_POS;if p>=rr.len() {terminate!();}let mut q=p;let mut ro=0usize;let mut io=0usize;let mut axis=re.rank();
    while axis>0 {axis-=1;let coordinate=q%re.shape(axis);q/=re.shape(axis);ro+=coordinate*re.stride(axis);
        if comptime!(complex) {io+=coordinate*im.stride(axis);}}
    let a=re[ro];if comptime!(complex) {let b=im[io];rr[p]=ar*a-ai*b;ri[p]=ar*b+ai*a;}else {rr[p]=ar*a;}
}
