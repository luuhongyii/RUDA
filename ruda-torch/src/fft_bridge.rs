//! PyTorch interleaved complex storage to exact GPU FFT plans.
use super::*;
use super::scientific::{real_tensor,complex_parts,copy_real,copy_complex};
use rufft::multidim::{NdFftPlan,NdFftKind,FftNormalization,reverse_normalization};
use rufft::spectral_layout as layout;
use ruda_core::tensor::Shape;
use ruda_kernel::tensor::RudaTensor;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;

type DeviceTensor=RudaTensor<CudaRuntime>;
fn norm(code:i64)->FftNormalization {
    match code {0=>FftNormalization::Backward,1=>FftNormalization::Forward,
        2=>FftNormalization::Ortho,3=>FftNormalization::None,_=>panic!("invalid FFT normalization")}
}
fn spec(xs:&[i64],rank:usize)->(FftNormalization,Vec<usize>,Vec<usize>) {
    assert!(xs.len()>=2);
    let k=usize::try_from(xs[1]).expect("negative FFT axis count");
    assert!(k>0 && xs.len()==2+2*k);
    let axes:Vec<usize>=xs[2..2+k].iter().map(|&a|usize::try_from(a).expect("negative FFT axis")).collect();
    let lengths:Vec<usize>=xs[2+k..].iter().map(|&n|usize::try_from(n).expect("negative FFT length")).collect();
    assert!(axes.iter().all(|&a|a<rank) && lengths.iter().all(|&n|n>0));
    assert!(!axes.iter().enumerate().any(|(i,a)|axes[..i].contains(a)));
    (norm(xs[0]),axes,lengths)
}
fn parts(x:&View)->(DeviceTensor,DeviceTensor) {
    if matches!(x.dtype,10|11|14) { complex_parts(x) }
    else {
        assert!(matches!(x.dtype,0|1|9),"FFT bridge expects F16/F32/F64 or complex storage");
        let real=real_tensor(x);
        let imag=layout::zeros(&real,real.meta.shape().clone()).expect("FFT imaginary allocation");
        (real,imag)
    }
}
fn transform_parts(x:&View,op:u32)->(DeviceTensor,Option<DeviceTensor>) {
    if op==2 || op==5 {
        assert!(matches!(x.dtype,0|1|9));
        (real_tensor(x),None)
    } else {
        let (re,im)=parts(x);(re,Some(im))
    }
}
fn write(re:&DeviceTensor,im:Option<&DeviceTensor>,out:&View) {
    if matches!(out.dtype,10|11|14) { copy_complex(re,im.expect("complex FFT imaginary component"),out); }
    else { assert!(matches!(out.dtype,0|1|9)); copy_real(re,out); }
}
fn output_dtype(input:u32,op:u32)->u32 {
    assert!(matches!(input,0|1|9|10|11|14));
    if matches!(input,1|14) { return if op==3 || op==4 {1} else {14}; }
    let double=input==9 || input==11;
    if op==3 || op==4 {if double {9} else {0}} else {if double {11} else {10}}
}
fn shift_bytes(input:&View,output:&View,axes:&[i64],inverse:bool) {
    assert_eq!(input.dtype,output.dtype);assert_eq!(input.shape,output.shape);
    let mut value=View {handle:input.handle.clone(),shape:input.shape.clone(),strides:input.strides.clone(),len:input.len,dtype:input.dtype};
    for &axis in axes {
        let axis=usize::try_from(axis).expect("negative shift axis");assert!(axis<input.shape.len());
        let next=View::packed(client().empty(value.len.checked_mul(element_bytes(value.dtype)).expect("shift byte size overflow").max(1)),value.shape.clone(),value.dtype);
        launch_byte_shift(&value,&next,axis,if inverse {value.shape[axis]/2} else {value.shape[axis]-value.shape[axis]/2});
        value=next;
    }
    launch_byte_shift(&value,output,0,0);
}
fn launch_byte_shift(input:&View,output:&View,axis:usize,shift:usize) {
    let input=primitives::tensor(&input.byte_view());let output=primitives::tensor(&output.byte_view());
    let count=output.meta.num_elements();
    if count>0 {
        assert!(count<=u32::MAX as usize,"shift exceeds U32 byte indexing");
        let dim=RudaDim::new(input.client.properties(),count);
        byte_shift::launch::<CudaRuntime>(&input.client,calculate_ruda_count_elemwise(&input.client,count,dim),dim,
            input.clone().into_tensor_arg(),output.into_tensor_arg(),axis,shift);
    }
}
fn transform(op:u32,re:&DeviceTensor,im:Option<&DeviceTensor>,axes:&[usize],lengths:&[usize],norm:FftNormalization)
    ->(DeviceTensor,Option<DeviceTensor>) {
    let kind=match op {0=>NdFftKind::ComplexForward,1=>NdFftKind::ComplexInverse,
        2|5=>NdFftKind::RealForward,3|4=>NdFftKind::RealInverse,_=>panic!("invalid FFT transform")};
    let actual_norm=if op==4 || op==5 {reverse_normalization(norm)} else {norm};
    let mut plan=NdFftPlan::new(re,axes,lengths,kind,actual_norm).expect("exact FFT plan");
    execute(&mut plan,op,re,im)
}
fn execute(plan:&mut NdFftPlan<CudaRuntime>,op:u32,re:&DeviceTensor,im:Option<&DeviceTensor>)->(DeviceTensor,Option<DeviceTensor>) {
    match op {
        0|1=>{
            let (re,im)=plan.execute_complex(re,im.expect("complex FFT input")).expect("complex FFT execution");
            (re,Some(im))
        },
        2|5=>{
            let (re,im)=plan.forward_real(re).expect("real FFT execution");
            if op==5 {(re,Some(layout::scale(&im,-1.0,None).expect("inverse Hermitian conjugation")))} else {(re,Some(im))}
        },
        3|4=>{
            let im=im.expect("inverse real FFT spectrum");
            let conjugate;
            let im=if op==4 {conjugate=layout::scale(im,-1.0,None).expect("Hermitian conjugation");&conjugate} else {im};
            let re=plan.inverse_real(re,im).expect("inverse real FFT execution");
            (re,None)
        },
        _=>unreachable!(),
    }
}
fn backward(op:u32,grad:&View,out:&View,axes:&[usize],lengths:&[usize],norm:FftNormalization) {
    let (mut re,mut im)=transform_parts(grad,if op==3 || op==4 {2} else {0});
    let last=*axes.last().unwrap(); let n=*lengths.last().unwrap();
    let reversed=reverse_normalization(norm);
    let (re,im)=match op {
        0|1=>transform(1-op,&re,im.as_ref(),axes,lengths,reversed),
        // The R2C adjoint uses a zero-extended one-sided spectrum, NOT a
        // mirrored spectrum. Taking the real part performs the real-input
        // projection without spuriously doubling the interior frequencies.
        2|5=>{
            let mut im=im.take().unwrap();
            if op==5 {im=layout::scale(&im,-1.0,None).expect("Hermitian adjoint conjugation");}
            let mut full=re.meta.shape().clone();
            for (&a,&s) in axes.iter().zip(lengths) {full[a]=s;}
            re=layout::resize(&re,full.clone()).expect("RFFT adjoint zero extension");
            im=layout::resize(&im,full).expect("RFFT adjoint zero extension");
            transform(1,&re,Some(&im),axes,lengths,if op==5 {norm} else {reversed})
        },
        3|4=>{
            let (re,im)=transform(2,&re,None,axes,lengths,if op==4 {norm} else {reversed});
            let re=layout::scale(&re,1.0,Some((last,n,2.0))).expect("IRFFT adjoint bins");
            let im=layout::scale(&im.unwrap(),if op==4 {-1.0} else {1.0},Some((last,n,2.0))).expect("IRFFT adjoint bins");
            (re,Some(im))
        },
        _=>panic!("invalid FFT adjoint"),
    };
    let re=layout::resize(&re,Shape::from(out.shape.clone())).expect("FFT padding adjoint");
    let im=if matches!(out.dtype,10|11|14) {
        Some(layout::resize(&im.expect("complex adjoint"),Shape::from(out.shape.clone())).expect("FFT padding adjoint"))
    } else {None};
    write(&re,im.as_ref(),out);
}

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_fft_api_version()->u32 {1}
/// Transform 0..5: C2C forward/inverse, R2C, C2R, HFFT, IHFFT.
/// Adjoint 20..25: same transform with descriptors [grad_output,grad_input].
/// Integer header [normalization,axis_count,axes...,exact_lengths...].
/// Shift 10/11: [input,output], ints=ordered axes. Frequencies 12/13:
/// [output], ints=[N], scalars=[spacing].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_fft(op:u32,ds:*const Descriptor,count:usize,
    ints:*const i64,ni:usize,scalars:*const f64,ns:usize)->i32 {
    checked(||{
        assert!(!ds.is_null() && count>0);
        let descriptors=unsafe {crate::abi::slice(ds,count)};
        let views:Vec<View>=descriptors.iter().map(|d|unsafe {crate::abi::view(d)}).collect();
        let args=if ni==0 {&[][..]} else {assert!(!ints.is_null());unsafe {crate::abi::slice(ints,ni)}};
        let floats=if ns==0 {&[][..]} else {assert!(!scalars.is_null());unsafe {crate::abi::slice(scalars,ns)}};
        if op<=5 || (20..=25).contains(&op) {
            assert_eq!(count,2); assert_eq!(ns,0);
            let (norm,axes,lengths)=spec(args,views[0].shape.len());
            if op>=20 {backward(op-20,&views[0],&views[1],&axes,&lengths,norm);}
            else {
                if op==2 || op==5 {assert!(matches!(views[0].dtype,0|1|9));}
                if op==3 || op==4 {assert!(matches!(views[0].dtype,0|1|9|10|11|14));}
                assert_eq!(views[1].dtype,output_dtype(views[0].dtype,op));
                let (re,im)=transform_parts(&views[0],op);
                let (re,im)=transform(op,&re,im.as_ref(),&axes,&lengths,norm);
                assert_eq!(re.meta.shape().as_slice(),views[1].shape.as_slice());
                write(&re,im.as_ref(),&views[1]);
            }
        } else if op==10 || op==11 {
            assert_eq!(count,2); assert_eq!(ns,0);
            shift_bytes(&views[0],&views[1],args,op==11);
        } else if op==12 || op==13 {
            assert_eq!(count,1); assert_eq!(ni,1); assert_eq!(ns,1);
            let n=usize::try_from(args[0]).expect("negative frequency length");
            let out=real_tensor(&views[0]);
            layout::frequencies(&out,n,floats[0],op==13).expect("FFT frequencies");
            copy_real(&out,&views[0]);
        } else if op==15 {
            assert_eq!(count,2);assert_eq!(ni,0);assert_eq!(ns,0);
            let (re,im)=parts(&views[0]);write(&re,Some(&im),&views[1]);
        } else {panic!("unknown FFT bridge operation");}
        LAUNCHES.fetch_add(1,Ordering::Relaxed);
        finish_dispatch(&client());
    })
}

#[ruda(launch)]
fn byte_shift(input:&Tensor<u8>,output:&mut Tensor<u8>,#[comptime] axis:usize,#[comptime] shift:usize) {
    let pos=ABSOLUTE_POS;if pos>=output.len() {terminate!();}
    let mut rem=pos;let mut read=0usize;let mut write=0usize;let mut dim=output.rank();
    while dim>0 {
        dim-=1;
        let c=rem%output.shape(dim);rem/=output.shape(dim);
        write+=c*output.stride(dim);
        let source=if dim==axis {(c+shift)%input.shape(dim)} else {c};
        read+=source*input.stride(dim);
    }
    output[write]=input[read];
}

struct StoredPlan {plan:NdFftPlan<CudaRuntime>,op:u32,dtype:u32,shape:Vec<usize>}
/// Explicit lifetime, no implicit plan cache. All calls on a handle must be
/// serialized by its owner and use its creation execution queue.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_fft_plan(op:u32,handle:*mut *mut std::ffi::c_void,
    ds:*const Descriptor,count:usize,ints:*const i64,ni:usize,result:*mut u64)->i32 {
    checked(||{
        assert!(!handle.is_null());
        if op==0 {
            assert!(unsafe {*handle}.is_null() && count==1 && ni>=3 && !ints.is_null() && !ds.is_null());
            let args=unsafe {crate::abi::slice(ints,ni)};
            let kind=u32::try_from(args[0]).expect("negative FFT plan kind"); assert!(kind<=5);
            let view=unsafe {crate::abi::view(ds)};
            if kind==2 || kind==5 {assert!(matches!(view.dtype,0|1|9));}
            let (norm,axes,lengths)=spec(&args[1..],view.shape.len());
            let (re,_)=transform_parts(&view,kind);
            let k=match kind {0=>NdFftKind::ComplexForward,1=>NdFftKind::ComplexInverse,
                2|5=>NdFftKind::RealForward,3|4=>NdFftKind::RealInverse,_=>unreachable!()};
            let plan=NdFftPlan::new(&re,&axes,&lengths,k,if kind==4 || kind==5 {reverse_normalization(norm)} else {norm})
                .expect("stored exact FFT plan");
            unsafe {*handle=Box::into_raw(Box::new(StoredPlan {plan,op:kind,dtype:view.dtype,shape:view.shape})).cast();}
        } else {
            assert!(!unsafe {*handle}.is_null());
            if op==4 {unsafe {drop(Box::from_raw((*handle).cast::<StoredPlan>()));*handle=std::ptr::null_mut();}return;}
            let saved=unsafe {&mut *((*handle).cast::<StoredPlan>())};
            match op {
                1=>{
                    assert!(count==2 && !ds.is_null());
                    let args=unsafe {crate::abi::slice(ds,count)};
                    let input=unsafe {crate::abi::view(&args[0])}; let output=unsafe {crate::abi::view(&args[1])};
                    assert_eq!(input.dtype,saved.dtype);assert_eq!(input.shape,saved.shape);
                    assert_eq!(output.dtype,output_dtype(input.dtype,saved.op));
                    let (re,im)=transform_parts(&input,saved.op);
                    let (re,im)=execute(&mut saved.plan,saved.op,&re,im.as_ref());
                    assert_eq!(re.meta.shape().as_slice(),output.shape.as_slice());write(&re,im.as_ref(),&output);
                    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&client());
                },
                2=>saved.plan.clear_lazy_workspace(),
                3=>{assert!(!result.is_null());unsafe {*result=saved.plan.retained_bytes() as u64;}},
                _=>panic!("unknown FFT plan command"),
            }
        }
    })
}
