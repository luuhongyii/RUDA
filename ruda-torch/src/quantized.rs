use super::*;
use ruda_kernel::{dsl as kernel_dsl, quantization::packed::{PackedTensor,PackedFormat}};

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_packed_api_version()->u32 { 1 }

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_mxfp8_api_version()->u32 { 1 }

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_fp4_api_version()->u32 { 1 }

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_packed_matmul_api_version()->u32 { 1 }

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_packed_indexed_groups_api_version()->u32 { 1 }

/// Six descriptors: x, codes, scales, offsets, group_index, destination.
/// Four layout words: format, group_size, logical_width, logical_rows.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_packed(op:u32,descriptors:*const Descriptor,count:usize,
    layout:*const u32,layout_count:usize)->i32 {
    checked(|| {
        assert_eq!(count,6); assert_eq!(layout_count,4);
        let ds=unsafe{abi::slice(descriptors,count)};
        let v:Vec<_>=ds.iter().map(|d|unsafe{abi::view(d)}).collect();
        let spec=unsafe{abi::slice(layout,layout_count)};
        let format=PackedFormat::from_code(spec[0]);
        let group=spec[1] as usize; let width=spec[2] as usize; let rows=spec[3] as usize;
        assert!(width>0 && group>0);
        if op==5 {
            let output=ruda_kernel::quantization::packed::sort_rows(&primitives::tensor(&v[0]));
            primitives::store(output,&v[5]);
        } else if op==0 {
            let input=primitives::tensor(&v[0]);
            assert_eq!(input.meta.shape().as_slice(),&[rows,width]);
            let result=PackedTensor::quantize(&input,primitives::tensor(&v[2]),primitives::tensor(&v[3]),format,group);
            primitives::store(result.codes,&v[5]);
        } else {
            let packed=PackedTensor::new(primitives::tensor(&v[1]),primitives::tensor(&v[2]),primitives::tensor(&v[3]),
                primitives::tensor(&v[4]),ruda_core::tensor::Shape::new([rows,width]),format,group);
            let result=match op {
                1=>packed.dequantize(primitives::tensor(&v[5]).dtype),
                2|3=>packed.linear(&primitives::tensor(&v[0]),op==3),
                6|7=>packed.linear_tensor_core(&primitives::tensor(&v[0]),op==7),
                4=>{
                    primitives::check_indices(&v[0],rows,false);
                    packed.embedding(&primitives::tensor(&v[0]),primitives::tensor(&v[5]).dtype)
                },
                _=>panic!("unknown packed quantization operation"),
            };
            primitives::store(result,&v[5]);
        }
        LAUNCHES.fetch_add(1,Ordering::Relaxed);
        finish_dispatch(&client());
    })
}

pub(super) fn copy_fp4(a:&View,out:&View) {
    assert!(a.dtype==16 && out.dtype==16,"packed FP4 copy does not perform numeric conversion");
    assert_eq!(a.shape,out.shape);
    crate::scientific::indexable(a);crate::scientific::indexable(out);
    if out.len==0 {return;}
    let c=client();let count=u32::try_from(out.len.div_ceil(128)).expect("FP4 copy grid overflow");
    unsafe {kernels::copy_bytes::launch::<CudaRuntime>(&c,RudaCount::Static(count,1,1),RudaDim::new_1d(128),a.byte_view().arg(),out.byte_view().arg());}
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);
}

#[ruda(launch)]
fn pack_nibbles<I:Size,O:Size>(input:&Tensor<u8>,output:&mut Tensor<u8>) {
    let p=ABSOLUTE_POS as usize;
    if p<output.len()/O::value() {
        let source=input.slice(0usize,input.len()).into_vectorized().with_vector_size::<I>();
        let bytes=source[p];let mut packed=Vector::<u8,O>::empty();
        #[unroll] for lane in 0usize..O::value() {packed[lane]=u8::cast_from(u32::cast_from(bytes[2*lane])+16u32*u32::cast_from(bytes[2*lane+1]));}
        let mut target=output.slice_mut(0usize,output.len()).into_vectorized().with_vector_size::<O>();target[p]=packed;
    }
}

pub(super) fn prepare_nibbles(c:&ComputeClient<CudaRuntime>,a:&View,out:&View)->PreparedKernel<CudaRuntime> {
    assert_eq!(a.dtype,8);assert_eq!(out.dtype,8);assert_eq!(a.len,2*out.len);
    assert!(a.contiguous() && out.contiguous());
    let width=[8usize,4,1].into_iter().find(|&width|out.len%width==0
        && a.handle.offset_start.unwrap_or(0)%(2*width) as u64==0 && out.handle.offset_start.unwrap_or(0)%width as u64==0).unwrap_or(1);
    macro_rules! run {($i:ty,$o:ty)=>{pack_nibbles::prepare::<CudaRuntime>(c,
        RudaCount::Static(u32::try_from((out.len/width).div_ceil(128)).expect("byte packing grid overflow"),1,1),
        RudaDim::new_1d(128),<$i as Size>::value(),<$o as Size>::value(),a.arg(),out.arg())};}
    match width {8=>run!(Const<16>,Const<8>),4=>run!(Const<8>,Const<4>),_=>run!(Const<2>,Const<1>)}
}

pub(super) fn convert(a:&View,out:&View) {
    if a.dtype==16 || out.dtype==16 {copy_fp4(a,out);return;}
    assert_eq!(a.shape,out.shape);
    assert!(matches!(a.dtype,12|13|15) || matches!(out.dtype,12|13|15));
    crate::scientific::indexable(a);crate::scientific::indexable(out);
    if out.len==0 {return;}
    if a.dtype!=out.dtype && vector_convert(a,out) {return;}
    let client=client();
    let count=u32::try_from(out.len.div_ceil(128)).expect("FP8 grid overflow");
    if a.dtype==out.dtype {
        unsafe { kernels::copy_bytes::launch::<CudaRuntime>(&client,RudaCount::Static(count,1,1),RudaDim::new_1d(128),a.byte_view().arg(),out.byte_view().arg()); }
    } else {
        let source_fp8=matches!(a.dtype,12|13|15);let destination_fp8=matches!(out.dtype,12|13|15);
        let real_input=matches!(a.dtype,10|11|14).then(||crate::scientific::component(a,false));
        let real_output=matches!(out.dtype,10|11|14).then(||crate::scientific::component(out,false));
        let imag_output=matches!(out.dtype,10|11|14).then(||crate::scientific::component(out,true));
        let input=real_input.as_ref().unwrap_or(a);let output=real_output.as_ref().unwrap_or(out);
        let imaginary=imag_output.as_ref().unwrap_or(output);
        crate::scientific::indexable(input);crate::scientific::indexable(output);crate::scientific::indexable(imaginary);
        macro_rules! from { ($ty:ty)=>{unsafe { from_fp8::launch::<$ty,CudaRuntime>(&client,RudaCount::Static(count,1,1),RudaDim::new_1d(128),input.arg(),output.arg(),imaginary.arg(),a.dtype,output.dtype,imag_output.is_some()); }}; }
        macro_rules! to { ($ty:ty)=>{unsafe { to_fp8::launch::<$ty,CudaRuntime>(&client,RudaCount::Static(count,1,1),RudaDim::new_1d(128),input.arg(),output.arg(),out.dtype); }}; }
        if source_fp8 && destination_fp8 {
            unsafe { between_fp8::launch::<CudaRuntime>(&client,RudaCount::Static(count,1,1),RudaDim::new_1d(128),a.arg(),out.arg(),a.dtype,out.dtype); }
        } else if source_fp8 { match output.dtype {
            0=>from!(f32),1=>from!(f16),2=>from!(bf16),3|8=>from!(u8),4=>from!(i64),5=>from!(i32),
            6=>from!(i16),7=>from!(i8),9=>from!(f64),_=>panic!("FP8 cast requires a real destination"),
        } } else { match input.dtype {
            0=>to!(f32),1=>to!(f16),2=>to!(bf16),3|8=>to!(u8),4=>to!(i64),5=>to!(i32),
            6=>to!(i16),7=>to!(i8),9=>to!(f64),_=>panic!("FP8 cast requires a real source"),
        } }
    }
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&client);
}

#[ruda(launch)]
fn convert_vector<I:Numeric+RudaElement,O:Numeric+RudaElement,V:Size>(input:&Tensor<I>,out:&mut Tensor<O>,
    #[comptime] source:u32,#[comptime] target:u32)
{
    let p=ABSOLUTE_POS as usize;
    if p<out.len()/V::value() {
        let packed=input.slice(0usize,input.len()).into_vectorized().with_vector_size::<V>();
        let values=packed[p];let mut converted=Vector::<O,V>::empty();
        #[unroll] for lane in 0usize..V::value() {
            let mut value=f32::cast_from(values[lane]);
            if comptime!(source==12 || source==13 || source==15) {value=crate::foreach_copy::decode_fp8(u32::cast_from(values[lane]),source);}
            if comptime!(target==12 || target==13 || target==15) {converted[lane]=O::cast_from(crate::foreach_copy::encode_fp8(value,target));}
            else {converted[lane]=O::cast_from(value);}
        }
        let mut target=out.slice_mut(0usize,out.len()).into_vectorized().with_vector_size::<V>();target[p]=converted;
    }
}

fn prepare_vector_convert(c:&ComputeClient<CudaRuntime>,a:&View,out:&View)->Option<PreparedKernel<CudaRuntime>> {
    if !matches!(a.dtype,0..=2|12|13|15) || !matches!(out.dtype,0..=2|12|13|15)
        || out.len%4!=0 || [a,out].iter().any(|v|!v.contiguous() || v.handle.offset_start.unwrap_or(0)%(4*element_bytes(v.dtype)) as u64!=0)
    {return None;}
    let vector=super::row_cached::memory_vector_size(c,out.len,&[a,out],16).max(4);
    let grid=RudaCount::Static(u32::try_from((out.len/vector).div_ceil(128)).expect("FP8 vector grid overflow"),1,1);
    macro_rules! input {($i:ty)=>{{
        macro_rules! run {($o:ty,$v:ty)=>{convert_vector::prepare::<$i,$o,CudaRuntime>(c,grid,RudaDim::new_1d(128),<$v as Size>::value(),a.arg(),out.arg(),a.dtype,out.dtype)};}
        macro_rules! output {($o:ty)=>{match vector {16=>run!($o,Const<16>),8=>run!($o,Const<8>),_=>run!($o,Const<4>)}};}
        match out.dtype {0=>output!(f32),1=>output!(f16),2=>output!(bf16),_=>output!(u8)}
    }};}
    Some(unsafe {match a.dtype {0=>input!(f32),1=>input!(f16),2=>input!(bf16),_=>input!(u8)}})
}

fn vector_convert(a:&View,out:&View)->bool {
    let c=client();
    let Some(kernel)=prepare_vector_convert(&c,a,out) else {return false;};
    let (task,count,args,origin)=kernel.into_parts();origin.launch(task,count,args);
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);true
}

pub(super) fn prepare_convert(c:&ComputeClient<CudaRuntime>,a:&View,out:&View)->PreparedKernel<CudaRuntime> {
    assert_eq!(a.shape,out.shape);
    assert_ne!(a.dtype,out.dtype);
    let source=matches!(a.dtype,12|13|15);let target=matches!(out.dtype,12|13|15);
    assert!(source || target);
    if let Some(kernel)=prepare_vector_convert(c,a,out) {return kernel;}
    let grid=RudaCount::Static(u32::try_from(out.len.div_ceil(128)).expect("FP8 cast grid overflow"),1,1);
    macro_rules! from {($t:ty)=>{unsafe {from_fp8::prepare::<$t,CudaRuntime>(c,grid,RudaDim::new_1d(128),
        a.arg(),out.arg(),out.arg(),a.dtype,out.dtype,false)}};}
    macro_rules! to {($t:ty)=>{unsafe {to_fp8::prepare::<$t,CudaRuntime>(c,grid,RudaDim::new_1d(128),a.arg(),out.arg(),out.dtype)}};}
    if source && target {
        unsafe {between_fp8::prepare::<CudaRuntime>(c,grid,RudaDim::new_1d(128),a.arg(),out.arg(),a.dtype,out.dtype)}
    } else if source {match out.dtype {
        0=>from!(f32),1=>from!(f16),2=>from!(bf16),3|8=>from!(u8),4=>from!(i64),5=>from!(i32),6=>from!(i16),7=>from!(i8),_=>unreachable!(),
    }} else {match a.dtype {
        0=>to!(f32),1=>to!(f16),2=>to!(bf16),3|8=>to!(u8),4=>to!(i64),5=>to!(i32),6=>to!(i16),7=>to!(i8),_=>unreachable!(),
    }}
}

#[ruda]
fn at<T:Numeric>(x:&Tensor<T>,i:usize)->usize {
    let mut remaining=i; let mut offset=0usize; let mut d=x.rank();
    while d>0 { d-=1; offset+=(remaining%x.shape(d))*x.stride(d);remaining/=x.shape(d); }
    offset
}
#[ruda(launch)]
fn from_fp8<T:Numeric+RudaElement>(x:&Tensor<u8>,out:&mut Tensor<T>,imag:&mut Tensor<T>,#[comptime] code:u32,#[comptime] destination:u32,#[comptime] complex:bool) {
    let i=ABSOLUTE_POS as usize;
    if i<out.len() {
        let value=crate::foreach_copy::decode_fp8(x[at(x,i)] as u32,code);
        let mut result=T::cast_from(value);
        if comptime!(destination==3) { result=T::cast_from(value!=0.0); }
        else if comptime!(destination==8) { result=T::cast_from(i64::cast_from(value)); }
        out[at(out,i)]=result;
        if comptime!(complex) {imag[at(imag,i)]=T::cast_from(0u32);}
    }
}
#[ruda(launch)]
fn to_fp8<T:Numeric+RudaElement>(x:&Tensor<T>,out:&mut Tensor<u8>,#[comptime] code:u32) {
    let i=ABSOLUTE_POS as usize;
    if i<out.len() { out[at(out,i)]=crate::foreach_copy::encode_fp8(f32::cast_from(x[at(x,i)]),code); }
}
#[ruda(launch)]
fn between_fp8(x:&Tensor<u8>,out:&mut Tensor<u8>,#[comptime] source:u32,#[comptime] target:u32) {
    let i=ABSOLUTE_POS as usize;
    if i<out.len() {
        out[at(out,i)]=crate::foreach_copy::encode_fp8(crate::foreach_copy::decode_fp8(x[at(x,i)] as u32,source),target);
    }
}

#[ruda(launch)]
fn sign_copy(x:&Tensor<u8>,out:&mut Tensor<u8>,#[comptime] negative:bool) {
    let i=ABSOLUTE_POS as usize;
    if i<out.len() {
        let mut value=x[at(x,i)];
        if comptime!(negative) {value^=0x80u8;}
        out[at(out,i)]=value;
    }
}

pub(super) fn resolve(a:&View,out:&View,negative:bool) {
    assert!(matches!(a.dtype,12|13|15));assert_eq!(a.dtype,out.dtype);assert_eq!(a.shape,out.shape);
    if a.dtype==15 { assert!(!negative,"E8M0 negative views are not supported");convert(a,out);return; }
    crate::scientific::indexable(a);crate::scientific::indexable(out);
    if out.len==0 {return;}
    let c=client();let count=u32::try_from(out.len.div_ceil(128)).expect("FP8 resolve grid overflow");
    unsafe {sign_copy::launch::<CudaRuntime>(&c,RudaCount::Static(count,1,1),RudaDim::new_1d(128),a.arg(),out.arg(),negative);}
    LAUNCHES.fetch_add(1,Ordering::Relaxed);
}
