use super::*;
use ruda_kernel::{dsl as kernel_dsl, quantization::packed::{PackedTensor,PackedFormat,decode_code,encode_code}};

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_packed_api_version()->u32 { 1 }

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

pub(super) fn convert(a:&View,out:&View) {
    assert_eq!(a.shape,out.shape);
    let client=client();
    let count=u32::try_from(out.len.div_ceil(128)).expect("FP8 grid overflow");
    if a.dtype==out.dtype {
        unsafe { kernels::copy_bytes::launch::<CudaRuntime>(&client,RudaCount::Static(count,1,1),RudaDim::new_1d(128),a.byte_view().arg(),out.byte_view().arg()); }
    } else {
        let format=if (if a.dtype>=12 {a.dtype} else {out.dtype})==12 {18} else {19};
        macro_rules! from { ($ty:ty)=>{unsafe { from_fp8::launch::<$ty,CudaRuntime>(&client,RudaCount::Static(count,1,1),RudaDim::new_1d(128),a.arg(),out.arg(),format,out.dtype==3); }}; }
        macro_rules! to { ($ty:ty)=>{unsafe { to_fp8::launch::<$ty,CudaRuntime>(&client,RudaCount::Static(count,1,1),RudaDim::new_1d(128),a.arg(),out.arg(),format); }}; }
        if a.dtype>=12 && out.dtype>=12 {
            unsafe { between_fp8::launch::<CudaRuntime>(&client,RudaCount::Static(count,1,1),RudaDim::new_1d(128),a.arg(),out.arg(),a.dtype==12); }
        } else if a.dtype>=12 { match out.dtype {
            0=>from!(f32),1=>from!(f16),2=>from!(bf16),3|8=>from!(u8),4=>from!(i64),5=>from!(i32),
            6=>from!(i16),7=>from!(i8),9=>from!(f64),_=>panic!("FP8 cast requires a real destination"),
        } } else { match a.dtype {
            0=>to!(f32),1=>to!(f16),2=>to!(bf16),3|8=>to!(u8),4=>to!(i64),5=>to!(i32),
            6=>to!(i16),7=>to!(i8),9=>to!(f64),_=>panic!("FP8 cast requires a real source"),
        } }
    }
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&client);
}

#[ruda]
fn at<T:Numeric>(x:&Tensor<T>,i:usize)->usize {
    let mut remaining=i; let mut offset=0usize; let mut d=x.rank();
    while d>0 { d-=1; offset+=(remaining%x.shape(d))*x.stride(d);remaining/=x.shape(d); }
    offset
}
#[ruda(launch)]
fn from_fp8<T:Numeric+RudaElement>(x:&Tensor<u8>,out:&mut Tensor<T>,#[comptime] format:u32,#[comptime] boolean:bool) {
    let i=ABSOLUTE_POS as usize;
    if i<out.len() {
        let value=decode_code(x[at(x,i)] as u32,format);
        let mut result=T::cast_from(value);
        if comptime!(boolean) { result=T::cast_from(value!=0.0); }
        out[at(out,i)]=result;
    }
}
#[ruda(launch)]
fn to_fp8<T:Numeric+RudaElement>(x:&Tensor<T>,out:&mut Tensor<u8>,#[comptime] format:u32) {
    let i=ABSOLUTE_POS as usize;
    if i<out.len() { out[at(out,i)]=encode_code(f32::cast_from(x[at(x,i)]),format,false) as u8; }
}
#[ruda(launch)]
fn between_fp8(x:&Tensor<u8>,out:&mut Tensor<u8>,#[comptime] e4m3:bool) {
    let i=ABSOLUTE_POS as usize;
    if i<out.len() {
        let source=comptime!(if e4m3 {18u32} else {19u32});
        let target=comptime!(if e4m3 {19u32} else {18u32});
        out[at(out,i)]=encode_code(decode_code(x[at(x,i)] as u32,source),target,false) as u8;
    }
}
