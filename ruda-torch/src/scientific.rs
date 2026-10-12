use super::*;
use ruda_core::tensor::{DType, Metadata};
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_contiguous_dtype};

pub(super) type Tensor = RudaTensor<CudaRuntime>;

fn storage_dtype(code: u32) -> DType {
    match code {
        0 | 10 => DType::F32, 1 => DType::F16, 2 => DType::BF16,
        3 | 8 => DType::U8, 4 => DType::I64, 5 => DType::I32,
        6 => DType::I16, 7 => DType::I8, 9 | 11 => DType::F64,
        _ => panic!("unsupported scientific dtype"),
    }
}
pub(super) fn real_tensor(v: &View) -> Tensor {
    assert!(v.dtype < 10, "a real storage descriptor is required");
    indexable(v);
    RudaTensor::new(client(), v.handle.clone(), Metadata::new(v.shape.clone(), v.strides.clone()),
        cuda_device(), storage_dtype(v.dtype))
}
fn component(v: &View, imaginary: bool) -> View {
    assert!(v.dtype == 10 || v.dtype == 11);
    let code = if v.dtype == 10 { 0 } else { 9 };
    let width = element_bytes(code);
    View { handle: v.handle.clone().offset_start(u64::from(imaginary) * width as u64),
        shape: v.shape.clone(), strides: v.strides.iter().map(|s| s.checked_mul(2).expect("complex stride overflow")).collect(),
        len: v.len, dtype: code }
}
fn allocate(shape: &[usize], dtype: DType) -> Tensor {
    empty_device_contiguous_dtype(client(), cuda_device(), shape.to_vec().into(), dtype)
}
fn tensor_view(t: &Tensor) -> View {
    let code = match t.dtype {
        DType::F32 => 0, DType::F16 => 1, DType::BF16 => 2, DType::I64 => 4,
        DType::I32 => 5, DType::I16 => 6, DType::I8 => 7, DType::U8 => 8, DType::F64 => 9,
        _ => panic!("unsupported scientific result dtype"),
    };
    View { handle: t.handle.clone(), shape: t.meta.shape().iter().copied().collect(),
        strides: t.meta.strides().iter().copied().collect(), len: t.meta.num_elements(), dtype: code }
}
pub(super) fn copy_real(src: &Tensor, out: &View) {
    assert_eq!(src.meta.num_elements(), out.len);
    assert_eq!(src.dtype, storage_dtype(out.dtype));
    assert!(out.dtype < 10);
    let source = tensor_view(src);
    copy_storage(&source, out);
}
pub(super) fn complex_parts(v: &View) -> (Tensor, Tensor) {
    let re = allocate(&v.shape, storage_dtype(v.dtype));
    let im = allocate(&v.shape, storage_dtype(v.dtype));
    copy_storage(&component(v, false), &tensor_view(&re));
    copy_storage(&component(v, true), &tensor_view(&im));
    (re, im)
}
pub(super) fn copy_complex(re: &Tensor, im: &Tensor, out: &View) {
    assert!(out.dtype == 10 || out.dtype == 11);
    assert_eq!(re.dtype, im.dtype); assert_eq!(re.dtype, storage_dtype(out.dtype));
    assert_eq!(re.meta.num_elements(), out.len); assert_eq!(im.meta.num_elements(), out.len);
    copy_storage(&tensor_view(re), &component(out, false));
    copy_storage(&tensor_view(im), &component(out, true));
}
fn grid(n: usize) -> RudaCount {
    assert!(n<=u32::MAX as usize,"scientific GPU work exceeds U32 indexing");
    RudaCount::Static(u32::try_from(n.div_ceil(128)).expect("scientific grid overflow"), 1, 1)
}
fn indexable(v:&View) {
    assert!(v.len<=u32::MAX as usize && v.shape.iter().all(|&d|d<=u32::MAX as usize),"scientific shape exceeds U32 indexing");
    if v.len!=0 {assert!(v.shape.iter().zip(&v.strides).try_fold(1usize,|n,(&d,&s)|
        (d-1).checked_mul(s).and_then(|term|n.checked_add(term))).is_some_and(|n|n<=u32::MAX as usize),
        "scientific strided span exceeds U32 indexing");}
}
fn copy_storage(a: &View, out: &View) {
    assert_eq!(a.len, out.len); assert_eq!(element_bytes(a.dtype), element_bytes(out.dtype));
    if out.len == 0 { return; }
    let bytes = out.len.checked_mul(element_bytes(out.dtype)).expect("scientific byte count overflow");
    indexable(&a.byte_view());indexable(&out.byte_view());
    unsafe { kernels::copy_bytes::launch::<CudaRuntime>(&client(), grid(bytes), RudaDim::new_1d(128),
        a.byte_view().arg(), out.byte_view().arg()); }
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
}
fn zero(v: &View) {
    if v.len == 0 { return; }
    let bytes = v.len.checked_mul(element_bytes(v.dtype)).expect("scientific byte count overflow");
    unsafe { kernels::fill_bytes::launch::<CudaRuntime>(&client(), grid(bytes), RudaDim::new_1d(128),
        v.byte_view().arg(), 0, 0); }
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
}
fn real_convert(a: &View, out: &View) {
    if out.len == 0 { return; }
    assert_eq!(a.len, out.len);
    if a.dtype == out.dtype { copy_storage(a, out); return; }
    macro_rules! run { ($i:ty, $o:ty) => {
        unsafe { cast_storage::launch::<$i, $o, CudaRuntime>(&client(), grid(out.len), RudaDim::new_1d(128),
            a.arg(), out.arg(), out.dtype == 3); }
    }; }
    macro_rules! destination { ($i:ty) => { match out.dtype {
        0 => run!($i, f32), 1 => run!($i, f16), 2 => run!($i, bf16), 3 | 8 => run!($i, u8),
        4 => run!($i, i64), 5 => run!($i, i32), 6 => run!($i, i16), 7 => run!($i, i8),
        9 => run!($i, f64), _ => panic!("real conversion destination required"),
    } }; }
    match a.dtype {
        0 => destination!(f32), 1 => destination!(f16), 2 => destination!(bf16), 3 | 8 => destination!(u8),
        4 => destination!(i64), 5 => destination!(i32), 6 => destination!(i16), 7 => destination!(i8),
        9 => destination!(f64), _ => panic!("real conversion source required"),
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
}
pub(super) fn convert(a: &View, out: &View) {
    assert_eq!(a.len, out.len);
    if a.dtype >= 10 {
        let re = component(a, false);
        if out.dtype >= 10 {
            real_convert(&re, &component(out, false));
            real_convert(&component(a, true), &component(out, true));
        } else if out.dtype == 3 {
            let im = component(a, true);
            unsafe {
                if a.dtype == 10 { complex_bool::launch::<f32, CudaRuntime>(&client(), grid(out.len), RudaDim::new_1d(128), re.arg(), im.arg(), out.arg()); }
                else { complex_bool::launch::<f64, CudaRuntime>(&client(), grid(out.len), RudaDim::new_1d(128), re.arg(), im.arg(), out.arg()); }
            }
            LAUNCHES.fetch_add(1, Ordering::Relaxed);
        } else { real_convert(&re, out); }
    } else if out.dtype >= 10 {
        real_convert(a, &component(out, false)); zero(&component(out, true));
    } else { real_convert(a, out); }
}
fn parts_as(v: &View, code: u32) -> (Tensor, Tensor) {
    let re = allocate(&v.shape, storage_dtype(code));
    let im = allocate(&v.shape, storage_dtype(code));
    if v.dtype >= 10 {
        real_convert(&component(v, false), &tensor_view(&re));
        real_convert(&component(v, true), &tensor_view(&im));
    } else { real_convert(v, &tensor_view(&re)); zero(&tensor_view(&im)); }
    (re, im)
}

pub(super) fn launch(op: u32, a: &View, b: &View, out: &View, scalar: f64) {
    indexable(a);indexable(b);indexable(out);
    if out.len == 0 { return; }
    if op == 0 { convert(a, out); return; }
    if op == 6 || op == 98 || op == 99 || op == 107 { reduce(a,out,op);return; }
    if op == 7 || op == 30 { matmul(a,b,out);return; }
    if op==97 {
        assert_eq!(a.shape,out.shape);assert_eq!(a.dtype,out.dtype);
        assert!(scalar>=0.0 && scalar.fract()==0.0 && (scalar as usize)<a.shape.len());
        let aa=a.byte_view();let oo=out.byte_view();indexable(&aa);indexable(&oo);
        unsafe {flip_bytes::launch::<CudaRuntime>(&client(),grid(oo.len),RudaDim::new_1d(128),aa.arg(),oo.arg(),scalar as u32);}
        LAUNCHES.fetch_add(1,Ordering::Relaxed);return;
    }
    if op==100 || op==101 {scan(a,out,scalar,op);return;}
    if op==84 {
        assert_eq!(a.dtype,3);assert_eq!(b.dtype,out.dtype);
        let bytes=out.len.checked_mul(element_bytes(out.dtype)).expect("select byte count overflow");
        unsafe {kernels::select_bytes::launch::<CudaRuntime>(&client(),grid(bytes),RudaDim::new_1d(128),a.arg(),b.byte_view().arg(),out.byte_view().arg());}
        LAUNCHES.fetch_add(1,Ordering::Relaxed);return;
    }
    if op==96 {integer_power(a,b,out);return;}
    assert!(matches!(op,1|2|3|4|8|9|10|11|12|13|14|17|19|20|21|22|23|24|25|26|27|28|29|35|36|37|38|39|40|49|50|57|78|79|80|81|82|83|116|117|118|121|122|123|124|206|208|214|216),"scientific operation is not implemented");
    let code = if a.dtype == 9 || a.dtype == 11 || b.dtype == 9 || b.dtype == 11 || out.dtype == 9 || out.dtype == 11 { 9 } else { 0 };
    let is_complex = a.dtype >= 10 || b.dtype >= 10 || out.dtype >= 10;
    assert!(!is_complex || matches!(op, 1 | 2 | 8 | 9 | 10 | 11 | 12 | 17 | 19 | 20 | 21 | 27 | 28 | 29 | 35 | 36 | 37 | 38 | 39 | 40 | 49 | 50 | 78 | 79 | 116 | 117 | 118 | 121 | 206 | 207 | 208 | 214), "operation is not defined for complex inputs");
    let (ar, ai) = parts_as(a, code); let (br, bi) = parts_as(b, code);
    if (78..=83).contains(&op) {
        assert_eq!(out.dtype, 3);
        unsafe {
            if code == 9 { compare::launch::<f64, CudaRuntime>(&client(), grid(out.len), RudaDim::new_1d(128), ar.into_tensor_arg(), ai.into_tensor_arg(), br.into_tensor_arg(), bi.into_tensor_arg(), out.arg(), op); }
            else { compare::launch::<f32, CudaRuntime>(&client(), grid(out.len), RudaDim::new_1d(128), ar.into_tensor_arg(), ai.into_tensor_arg(), br.into_tensor_arg(), bi.into_tensor_arg(), out.arg(), op); }
        }
    } else {
        let re = allocate(&out.shape, storage_dtype(code)); let im = allocate(&out.shape, storage_dtype(code));
        let (oldr,oldi)=if op==28 || op==29 || op==49 {parts_as(out,code)} else {(ar.clone(),ai.clone())};
        unsafe {
            if code == 9 { pointwise::launch::<f64, CudaRuntime>(&client(), grid(out.len), RudaDim::new_1d(128), ar.into_tensor_arg(), ai.into_tensor_arg(), br.into_tensor_arg(), bi.into_tensor_arg(), oldr.into_tensor_arg(), oldi.into_tensor_arg(), re.clone().into_tensor_arg(), im.clone().into_tensor_arg(), scalar, op, is_complex); }
            else { pointwise::launch::<f32, CudaRuntime>(&client(), grid(out.len), RudaDim::new_1d(128), ar.into_tensor_arg(), ai.into_tensor_arg(), br.into_tensor_arg(), bi.into_tensor_arg(), oldr.into_tensor_arg(), oldi.into_tensor_arg(), re.clone().into_tensor_arg(), im.clone().into_tensor_arg(), scalar as f32, op, is_complex); }
        }
        if out.dtype >= 10 { copy_complex(&re, &im, out); }
        else { real_convert(&tensor_view(&re), out); }
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
}

fn matmul(a:&View,b:&View,out:&View) {
    use rublas::tensor_matmul::{gemm,GemmOptions,MatmulStrategy};
    let multiply=|a:Tensor,b:Tensor,c:Option<Tensor>,beta:f64| gemm(a,b,c,GemmOptions{beta,..Default::default()},MatmulStrategy::Auto).expect("scientific GEMM failed");
    if a.dtype>=10 || b.dtype>=10 || out.dtype>=10 {
        let code=if out.dtype==11 {9} else {0};
        let (ar,ai)=parts_as(a,code);let(br,bi)=parts_as(b,code);
        let real=multiply(ar.clone(),br.clone(),Some(multiply(ai.clone(),bi.clone(),None,0.0)),-1.0);
        let imag=multiply(ar,bi,Some(multiply(ai,br,None,0.0)),1.0);
        copy_complex(&real,&imag,out);
    } else {
        assert_eq!(a.dtype,b.dtype);assert!(a.dtype==0 || a.dtype==9);
        copy_real(&multiply(real_tensor(a),real_tensor(b),None,0.0),out);
    }
}

fn reduce(a: &View, out: &View, operation: u32) {
    let code = if a.dtype == 9 || a.dtype == 11 { 9 } else { 0 };
    let (ar, ai) = parts_as(a, code);
    let re = allocate(&out.shape, storage_dtype(code)); let im = allocate(&out.shape, storage_dtype(code));
    assert_eq!(a.shape.len(), out.shape.len());
    assert!(a.shape.iter().zip(&out.shape).all(|(a,b)| a==b || *b==1));
    if out.len != 0 {
        unsafe {
            if code == 9 { reduction::launch::<f64, CudaRuntime>(&client(), grid(out.len), RudaDim::new_1d(128), ar.into_tensor_arg(), ai.into_tensor_arg(), re.clone().into_tensor_arg(), im.clone().into_tensor_arg(), operation); }
            else { reduction::launch::<f32, CudaRuntime>(&client(), grid(out.len), RudaDim::new_1d(128), ar.into_tensor_arg(), ai.into_tensor_arg(), re.clone().into_tensor_arg(), im.clone().into_tensor_arg(), operation); }
        }
    }
    if out.dtype >= 10 { copy_complex(&re,&im,out); } else { real_convert(&tensor_view(&re),out); }
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
}

fn integer_power(a:&View,b:&View,out:&View) {
    assert!((4..=7).contains(&b.dtype),"integer power exponent requires signed integer storage");
    assert_eq!(a.shape,b.shape);assert_eq!(a.shape,out.shape);assert_eq!(a.dtype,out.dtype);
    let code=if a.dtype==9 || a.dtype==11 {9}else {0};let complex=a.dtype>=10;
    let (ar,ai)=parts_as(a,code);let exponent=real_tensor(b);
    let re=allocate(&out.shape,storage_dtype(code));let im=allocate(&out.shape,storage_dtype(code));
    unsafe {
        if code==9 {integer_power_kernel::launch::<f64,CudaRuntime>(&client(),grid(out.len),RudaDim::new_1d(128),ar.into_tensor_arg(),ai.into_tensor_arg(),exponent.clone().into_tensor_arg(),re.clone().into_tensor_arg(),im.clone().into_tensor_arg(),complex,exponent.dtype.into());}
        else {integer_power_kernel::launch::<f32,CudaRuntime>(&client(),grid(out.len),RudaDim::new_1d(128),ar.into_tensor_arg(),ai.into_tensor_arg(),exponent.clone().into_tensor_arg(),re.clone().into_tensor_arg(),im.clone().into_tensor_arg(),complex,exponent.dtype.into());}
    }
    if complex {copy_complex(&re,&im,out);}else {real_convert(&tensor_view(&re),out);}
    LAUNCHES.fetch_add(1,Ordering::Relaxed);
}
fn scan(a:&View,out:&View,axis:f64,op:u32) {
    assert_eq!(a.shape,out.shape);assert_eq!(a.dtype,out.dtype);
    assert!(axis>=0.0 && axis.fract()==0.0 && (axis as usize)<a.shape.len());
    let code=if a.dtype==9 || a.dtype==11 {9}else {0};let complex=a.dtype>=10;
    let (ar,ai)=parts_as(a,code);let re=allocate(&out.shape,storage_dtype(code));let im=allocate(&out.shape,storage_dtype(code));
    unsafe {
        if code==9 {scan_kernel::launch::<f64,CudaRuntime>(&client(),grid(out.len),RudaDim::new_1d(128),ar.into_tensor_arg(),ai.into_tensor_arg(),re.clone().into_tensor_arg(),im.clone().into_tensor_arg(),axis as u32,op==101);}
        else {scan_kernel::launch::<f32,CudaRuntime>(&client(),grid(out.len),RudaDim::new_1d(128),ar.into_tensor_arg(),ai.into_tensor_arg(),re.clone().into_tensor_arg(),im.clone().into_tensor_arg(),axis as u32,op==101);}
    }
    if complex {copy_complex(&re,&im,out);}else {real_convert(&tensor_view(&re),out);}
    LAUNCHES.fetch_add(1,Ordering::Relaxed);
}

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_scientific_api_version() -> u32 { 1 }

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_scientific(op: u32, descriptors: *const Descriptor, count: usize,
    integers: *const i64, integer_count: usize, scalars: *const f64, scalar_count: usize) -> i32 {
    checked(|| {
        assert!(!descriptors.is_null());
        assert!(integer_count == 0 || !integers.is_null()); assert!(scalar_count == 0 || !scalars.is_null());
        let xs: Vec<View> = unsafe { std::slice::from_raw_parts(descriptors,count) }.iter().map(|d| unsafe { View::read(d) }).collect();
        for x in &xs {indexable(x);}
        let ints = if integer_count == 0 { &[][..] } else { unsafe { std::slice::from_raw_parts(integers,integer_count) } };
        let values = if scalar_count == 0 { &[][..] } else { unsafe { std::slice::from_raw_parts(scalars,scalar_count) } };
        match op {
            6 | 98 | 99 | 107 => { assert_eq!(count,2); reduce(&xs[0],&xs[1],op); }
            201 => {
                assert_eq!(count,2);assert_eq!(ints.len(),2);assert!(xs[0].shape.len()>=2);assert_eq!(xs[0].shape,xs[1].shape);
                let bytes=xs[1].len.checked_mul(element_bytes(xs[1].dtype)).expect("triangular byte count overflow");
                if bytes!=0 {unsafe {triangular_bytes::launch::<CudaRuntime>(&client(),grid(bytes),RudaDim::new_1d(128),xs[0].byte_view().arg(),xs[1].byte_view().arg(),ints[0],ints[1]!=0);}}
            }
            205 => {
                assert_eq!(count,1); assert_eq!(values.len(),2);
                let out = &xs[0]; let code = if out.dtype==11 || out.dtype==9 {9} else {0};
                let re = allocate(&out.shape,storage_dtype(code)); let im = allocate(&out.shape,storage_dtype(code));
                if out.len != 0 { unsafe {
                    if code==9 { constant::launch::<f64,CudaRuntime>(&client(),grid(out.len),RudaDim::new_1d(128),re.clone().into_tensor_arg(),im.clone().into_tensor_arg(),values[0],values[1]); }
                    else { constant::launch::<f32,CudaRuntime>(&client(),grid(out.len),RudaDim::new_1d(128),re.clone().into_tensor_arg(),im.clone().into_tensor_arg(),values[0] as f32,values[1] as f32); }
                } }
                if out.dtype>=10 {copy_complex(&re,&im,out);} else {real_convert(&tensor_view(&re),out);}
            }
            207 => {
                assert_eq!(count,2); assert_eq!(ints.len(),2);
                if (3..=8).contains(&xs[0].dtype) {
                    assert_eq!(xs[0].dtype,xs[1].dtype);assert_eq!(xs[0].len,xs[1].len);
                    if ints[1]==0 {copy_storage(&xs[0],&xs[1]);}
                    else {
                        assert_ne!(xs[0].dtype,3,"boolean negative views are not defined");
                        macro_rules! run {($i:ty)=>{unsafe {negative_integer::launch::<$i,CudaRuntime>(&client(),grid(xs[1].len),RudaDim::new_1d(128),xs[0].arg(),xs[1].arg());}}}
                        if xs[1].len!=0 {match xs[0].dtype {4=>run!(i64),5=>run!(i32),6=>run!(i16),7=>run!(i8),8=>run!(u8),_=>unreachable!()}}
                    }
                }else {
                    let (re,im)=parts_as(&xs[0],if xs[1].dtype==9 || xs[1].dtype==11 {9} else {0});
                    let negative=if ints[1]!=0 {-1.0} else {1.0};
                    let imaginary=if ints[0]!=0 {-negative} else {negative};
                    scale_copy(&re,&xs[1],false,negative);
                    if xs[1].dtype>=10 {scale_copy(&im,&xs[1],true,imaginary);}
                }
            }
            _ => { assert_eq!(count,3); launch(op,&xs[0],&xs[1],&xs[2],values.first().copied().unwrap_or(1.0)); }
        }
        finish_dispatch(&client());
    })
}

fn scale_copy(a: &Tensor, out: &View, imaginary: bool, factor: f64) {
    let destination = if out.dtype>=10 {component(out,imaginary)} else {View{handle:out.handle.clone(),shape:out.shape.clone(),strides:out.strides.clone(),len:out.len,dtype:out.dtype}};
    if out.len==0 {return;}
    let scaled=allocate(&out.shape,a.dtype);
    unsafe {
        if a.dtype==DType::F64 {scale::launch::<f64,CudaRuntime>(&client(),grid(out.len),RudaDim::new_1d(128),a.clone().into_tensor_arg(),scaled.clone().into_tensor_arg(),factor);}
        else {scale::launch::<f32,CudaRuntime>(&client(),grid(out.len),RudaDim::new_1d(128),a.clone().into_tensor_arg(),scaled.clone().into_tensor_arg(),factor as f32);}
    }
    real_convert(&tensor_view(&scaled),&destination);
}

#[ruda]
fn address<T: Numeric>(x: &kernel_dsl::prelude::Tensor<T>, position: usize) -> usize {
    let mut remaining=position; let mut offset=0usize; let mut axis=x.rank();
    while axis>0 {axis-=1;offset+=(remaining%x.shape(axis))*x.stride(axis);remaining/=x.shape(axis);}
    offset
}
#[ruda(launch)]
fn cast_storage<I: Numeric+RudaElement,O: Numeric+RudaElement>(a:&kernel_dsl::prelude::Tensor<I>,out:&mut kernel_dsl::prelude::Tensor<O>,#[comptime] boolean:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {let x=a[address(a,p)];let mut value=O::cast_from(x);if comptime!(boolean) {value=O::cast_from(x!=I::cast_from(0u32));}out[address(out,p)]=value;}
}
#[ruda(launch)]
fn negative_integer<I:Numeric+RudaElement>(a:&kernel_dsl::prelude::Tensor<I>,out:&mut kernel_dsl::prelude::Tensor<I>) {
    let p=ABSOLUTE_POS as usize;if p<out.len() {out[address(out,p)]=I::cast_from(0u32)-a[address(a,p)];}
}
#[ruda(launch)]
fn complex_bool<F:Float>(re:&kernel_dsl::prelude::Tensor<F>,im:&kernel_dsl::prelude::Tensor<F>,out:&mut kernel_dsl::prelude::Tensor<u8>) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {out[address(out,p)]=u8::cast_from(re[address(re,p)]!=F::new(0.0)||im[address(im,p)]!=F::new(0.0));}
}
#[ruda(launch)]
fn scale<F:Float>(a:&kernel_dsl::prelude::Tensor<F>,out:&mut kernel_dsl::prelude::Tensor<F>,factor:F) {
    let p=ABSOLUTE_POS as usize;if p<out.len() {out[address(out,p)]=a[address(a,p)]*factor;}
}
#[ruda(launch)]
fn constant<F:Float>(re:&mut kernel_dsl::prelude::Tensor<F>,im:&mut kernel_dsl::prelude::Tensor<F>,r:F,i:F) {
    let p=ABSOLUTE_POS as usize;if p<re.len() {re[address(re,p)]=r;im[address(im,p)]=i;}
}
#[ruda]
fn signed<F:Float>(magnitude:F,sign:F)->F {
    let mut result=magnitude.abs();
    if sign<F::new(0.0) || (sign==F::new(0.0) && F::new(1.0)/sign<F::new(0.0)) {result=-result;}
    result
}
#[ruda]
fn complex_log<F:Float>(x:F,y:F)->(F,F) {
    let mut scale=x.abs();if y.abs()>scale {scale=y.abs();}
    let mut real=scale.ln();
    if scale!=F::new(0.0) && !scale.is_inf() {
        let a=x/scale;let b=y/scale;real+=(a*a+b*b).ln()/F::new(2.0);
    }
    (real,y.atan2(x))
}
#[ruda]
fn complex_sqrt<F:Float>(x:F,y:F)->(F,F) {
    let z=F::new(0.0);let two=F::new(2.0);let mut r=F::NAN;let mut i=F::NAN;
    if y.is_inf() {r=F::INFINITY;i=y;}
    else if x.is_inf() {
        if x>z {r=x;i=signed(z,y);if y.is_nan() {i=y;}}
        else {r=z;i=signed(F::INFINITY,y);if y.is_nan() {r=y;}}
    }
    else if !x.is_nan() && !y.is_nan() {
        let ax=x.abs();let ay=y.abs();let mut scale=ax;if ay>scale {scale=ay;}
        if scale==z {r=z;i=y;}
        else {
            let magnitude=(x/scale).hypot(y/scale);
            let t=scale.sqrt()*((magnitude+ax/scale)/two).sqrt();
            if x>=z {r=t;i=(y/t)/two;}
            else {r=(ay/t)/two;i=signed(t,y);}
        }
    }
    (r,i)
}
#[ruda]
fn inverse_hyperbolic<F:Float>(x:F,y:F,#[comptime] acosh:bool)->(F,F) {
    let one=F::new(1.0);let ax=x.abs();let ay=y.abs();
    let mut scale=ax;if ay>scale {scale=ay;}if scale<one {scale=one;}
    let mut xx=ax/scale;let yy=ay/scale;let epsilon=one/scale;
    if comptime!(acosh) {xx=x/scale;}
    let mut ar=F::new(0.0);let mut ai=F::new(0.0);let mut br=F::new(0.0);let mut bi=F::new(0.0);
    if comptime!(acosh) {let (a,b)=complex_sqrt(xx+epsilon,yy);ar=a;ai=b;let (c,d)=complex_sqrt(xx-epsilon,yy);br=c;bi=d;}
    else {let (a,b)=complex_sqrt(xx,yy+epsilon);ar=a;ai=b;let (c,d)=complex_sqrt(xx,yy-epsilon);br=c;bi=d;}
    let sr=ar*br-ai*bi;let si=ar*bi+ai*br;
    let (lr,li)=complex_log(xx+sr,yy+si);
    let mut r=lr+scale.ln();let mut i=signed(li,y);
    if comptime!(!acosh) {
        r=signed(r,x);
        if ax<F::new(0.5) && ay<F::new(0.5) {
            let s=ax*ax+ay*ay-one;
            let t=F::new(2.0).sqrt()*ax/(s.hypot(F::new(2.0)*ax)-s).sqrt();
            r=signed(t.asinh(),x);i=(y/r.cosh()).asin();
        }
        if y==F::new(0.0) {r=x.asinh();i=y;}
    }
    if scale.is_inf() {
        r=F::INFINITY;
        let mut angle=y.atan2(x);
        if comptime!(!acosh) {angle=y.atan2(ax);r=signed(r,x);}
        i=angle;
    }
    (r,i)
}
#[ruda(launch)]
fn pointwise<F:Float>(ar:&kernel_dsl::prelude::Tensor<F>,ai:&kernel_dsl::prelude::Tensor<F>,br:&kernel_dsl::prelude::Tensor<F>,bi:&kernel_dsl::prelude::Tensor<F>,oldr:&kernel_dsl::prelude::Tensor<F>,oldi:&kernel_dsl::prelude::Tensor<F>,re:&mut kernel_dsl::prelude::Tensor<F>,im:&mut kernel_dsl::prelude::Tensor<F>,alpha:F,#[comptime] op:u32,#[comptime] complex:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<re.len() {
        let x=ar[address(ar,p)];let u=ai[address(ai,p)];let y=br[address(br,p)];let v=bi[address(bi,p)];
        let mut r=x;let mut i=u;let z=F::new(0.0);let one=F::new(1.0);
        if comptime!(op==1 || op==116) {r=x+alpha*y;i=u+alpha*v;}
        else if comptime!(op==2 || op==117 || op==28) {r=x*y-u*v;i=x*v+u*y;}
        else if comptime!(op==8 || op==118 || op==27 || op==29) {
            let mut nr=x;let mut ni=u;let mut dr=y;let mut di=v;
            if comptime!(op==27) {nr=one;ni=z;dr=x;di=u;}
            if di.abs()>dr.abs() {let ratio=dr/di;let den=di+dr*ratio;r=(nr*ratio+ni)/den;i=(ni*ratio-nr)/den;}
            else {let ratio=di/dr;let den=dr+di*ratio;r=(nr+ni*ratio)/den;i=(ni-nr*ratio)/den;}
            if comptime!(!complex) {r=nr/dr;i=z;}
        }
        else if comptime!(op==208) {r=-x;i=-u;}
        else if comptime!(op==206) {i=-u;}
        else if comptime!(op==3) {if x<z {r=z;}}
        else if comptime!(op==4) {if y<=alpha {r=z;}}
        else if comptime!(op==9) {let magnitude=x.exp();r=magnitude*u.cos();i=magnitude*u.sin();}
        else if comptime!(op==10) {if comptime!(complex) {let (a,b)=complex_log(x,u);r=a;i=b;}else {r=x.ln();i=z;}}
        else if comptime!(op==11 || op==12) {
            if comptime!(complex) {let (a,b)=complex_sqrt(x,u);r=a;i=b;
                if comptime!(op==12) {let rr=r;let ii=i;
                    if ii.abs()>rr.abs() {let ratio=rr/ii;let den=ii+rr*ratio;r=ratio/den;i=-one/den;}
                    else {let ratio=ii/rr;let den=rr+ii*ratio;r=one/den;i=-ratio/den;}
                }
            } else {r=x.sqrt();i=z;if comptime!(op==12) {r=one/r;}}
        }
        else if comptime!(op==13) {r=one/(one+(-x).exp());i=z;}
        else if comptime!(op==14) {r=x/(one+(-x).exp());i=z;}
        else if comptime!(op==17) {r=x.tanh();i=z;if comptime!(complex) {let den=(F::new(2.0)*x).cosh()+(F::new(2.0)*u).cos();r=(F::new(2.0)*x).sinh()/den;i=(F::new(2.0)*u).sin()/den;}}
        else if comptime!(op==19) {r=x.sin();i=z;if comptime!(complex) {r*=u.cosh();i=x.cos()*u.sinh();}}
        else if comptime!(op==20) {r=x.cos();i=z;if comptime!(complex) {r*=u.cosh();i=-x.sin()*u.sinh();}}
        else if comptime!(op==21) {r=x.hypot(u);i=z;}
        else if comptime!(op==22) {r=z;if x>z {r=one;}else if x<z {r=-one;}}
        else if comptime!(op==23) {r=x.floor();}
        else if comptime!(op==24) {r=x.ceil();}
        else if comptime!(op==25) {r=x.trunc();}
        else if comptime!(op==26) {r=x.round();}
        else if comptime!(op==35) {r=x.log1p();i=z;if comptime!(complex) {
            let (a,b)=complex_log(one+x,u);r=a;i=b;
            if x.abs()<F::new(0.5) && u.abs()<F::new(0.5) {r=(F::new(2.0)*x+x*x+u*u).log1p()/F::new(2.0);}
        }}
        else if comptime!(op==36) {r=x.sinh();i=z;if comptime!(complex) {r*=u.cos();i=x.cosh()*u.sin();}}
        else if comptime!(op==37) {r=x.cosh();i=z;if comptime!(complex) {r*=u.cos();i=x.sinh()*u.sin();}}
        else if comptime!(op==38) {r=x.asinh();i=z;if comptime!(complex) {let (a,b)=inverse_hyperbolic(x,u,false);r=a;i=b;}}
        else if comptime!(op==39) {r=x.acosh();i=z;if comptime!(complex) {let (a,b)=inverse_hyperbolic(x,u,true);r=a;i=b;}}
        else if comptime!(op==40) {r=x.atanh();i=z;if comptime!(complex) {
            let (lr,_li)=complex_log(one+x,u);let (rr,_ri)=complex_log(one-x,u);
            r=(lr-rr)/F::new(2.0);i=(u.atan2(one+x)+u.atan2(one-x))/F::new(2.0);
            if x.abs()<F::new(0.5) {
                let mut scale=u.abs();if scale<one {scale=one;}
                let a=(one-x)/scale;let b=u/scale;
                r=(((F::new(4.0)*x/scale)/scale)/(a*a+b*b)).log1p()/F::new(4.0);
            }
            if x==z {r=x;}if x.is_inf() || u.is_inf() {r=signed(z,x);i=signed(one.atan2(z),u);}
        }}
        else if comptime!(op==49) {
            let start=oldr[address(oldr,p)];let start_i=oldi[address(oldi,p)];
            let dr=x-start;let di=u-start_i;let mut wr=y;let mut wi=v;let mut base=start;let mut base_i=start_i;
            if y.hypot(v)>=F::new(0.5) {wr-=one;base=x;base_i=u;}
            r=base+wr*dr-wi*di;i=base_i+wr*di+wi*dr;
        }
        else if comptime!(op==50) {r=y-alpha*x;i=v-alpha*u;}
        else if comptime!(op==216) {r=x.erf();i=z;}
        else if comptime!(op==121) {r=x.powf(y);if comptime!(complex) {let logmag=x.hypot(u).ln();let angle=u.atan2(x);let mag=(y*logmag-v*angle).exp();let phase=y*angle+v*logmag;r=mag*phase.cos();i=mag*phase.sin();if y==z && v==z {r=one;i=z;}}}
        else if comptime!(op==57 || op==122) {r=(x/y).trunc();if comptime!(op==122) {r=(x/y).floor();}}
        else if comptime!(op==123 || op==124) {r=x%y;if comptime!(op==124) {if r!=z && ((r<z)!=(y<z)) {r+=y;}}}
        else if comptime!(op==214) {let magnitude=x.hypot(u);r=z;i=z;if magnitude!=z {r=x/magnitude;i=u/magnitude;}}
        if comptime!(op==28 || op==29) {r=oldr[address(oldr,p)]+alpha*r;i=oldi[address(oldi,p)]+alpha*i;}
        re[address(re,p)]=r;im[address(im,p)]=i;
    }
}
#[ruda(launch)]
fn compare<F:Float>(ar:&kernel_dsl::prelude::Tensor<F>,ai:&kernel_dsl::prelude::Tensor<F>,br:&kernel_dsl::prelude::Tensor<F>,bi:&kernel_dsl::prelude::Tensor<F>,out:&mut kernel_dsl::prelude::Tensor<u8>,#[comptime] op:u32) {
    let p=ABSOLUTE_POS as usize;if p<out.len() {let x=ar[address(ar,p)];let y=br[address(br,p)];let equal=x==y && ai[address(ai,p)]==bi[address(bi,p)];let mut result=equal;
        if comptime!(op==79) {result=!equal;}else if comptime!(op==80) {result=x<y;}else if comptime!(op==81) {result=x<=y;}else if comptime!(op==82) {result=x>y;}else if comptime!(op==83) {result=x>=y;}
        out[address(out,p)]=u8::cast_from(result);}
}
#[ruda(launch)]
fn integer_power_kernel<F:Float,I:Int>(ar:&kernel_dsl::prelude::Tensor<F>,ai:&kernel_dsl::prelude::Tensor<F>,exponent:&kernel_dsl::prelude::Tensor<I>,re:&mut kernel_dsl::prelude::Tensor<F>,im:&mut kernel_dsl::prelude::Tensor<F>,#[comptime] complex:bool,#[define(I)] _dtype:StorageType) {
    let p=ABSOLUTE_POS as usize;if p>=re.len() {terminate!();}
    let e=i64::cast_from(exponent[address(exponent,p)]);let negative=e<0i64;
    let mut n=u64::cast_from(e);if negative {n=u64::cast_from(-(e+1i64))+1u64;}
    let mut xr=ar[address(ar,p)];let mut xi=ai[address(ai,p)];
    let mut r=F::new(1.0);let mut i=F::new(0.0);
    if negative {
        if comptime!(complex) {
            if xi.abs()>xr.abs() {let ratio=xr/xi;let den=xi+xr*ratio;xr=ratio/den;xi=-F::new(1.0)/den;}
            else {let ratio=xi/xr;let den=xr+xi*ratio;xr=F::new(1.0)/den;xi=-ratio/den;}
        }else {xr=F::new(1.0)/xr;}
    }
    while n!=0u64 {
        if (n&1u64)!=0u64 {
            if comptime!(complex) {let rr=r*xr-i*xi;let ii=r*xi+i*xr;r=rr;i=ii;}
            else {r*=xr;}
        }
        n>>=1u64;
        if n!=0u64 {
            if comptime!(complex) {let rr=xr*xr-xi*xi;let ii=F::new(2.0)*xr*xi;xr=rr;xi=ii;}
            else {xr*=xr;}
        }
    }
    re[address(re,p)]=r;im[address(im,p)]=i;
}
#[ruda(launch)]
fn flip_bytes(a:&kernel_dsl::prelude::Tensor<u8>,out:&mut kernel_dsl::prelude::Tensor<u8>,axis:u32) {
    let p=ABSOLUTE_POS as usize;if p>=out.len() {terminate!();}
    let mut q=p;let mut source=0usize;let mut d=out.rank();
    while d>0 {d-=1;let mut coordinate=q%out.shape(d);q/=out.shape(d);
        if d==axis as usize {coordinate=out.shape(d)-1-coordinate;}
        source+=coordinate*a.stride(d);
    }out[address(out,p)]=a[source];
}
#[ruda(launch)]
fn scan_kernel<F:Float>(ar:&kernel_dsl::prelude::Tensor<F>,ai:&kernel_dsl::prelude::Tensor<F>,re:&mut kernel_dsl::prelude::Tensor<F>,im:&mut kernel_dsl::prelude::Tensor<F>,axis:u32,#[comptime] product:bool) {
    let p=ABSOLUTE_POS as usize;if p>=re.len() {terminate!();}
    let mut q=p;let mut base=0usize;let mut last=0usize;let mut d=re.rank();
    while d>0 {d-=1;let coordinate=q%re.shape(d);q/=re.shape(d);
        if d==axis as usize {last=coordinate;}else {base+=coordinate*ar.stride(d);}
    }
    let mut r=F::new(0.0);let mut i=F::new(0.0);if comptime!(product) {r=F::new(1.0);}
    let mut j=0usize;while j<=last {
        let pos=base+j*ar.stride(axis as usize);let x=ar[pos];let y=ai[pos];
        if comptime!(product) {let rr=r*x-i*y;let ii=r*y+i*x;r=rr;i=ii;}else {r+=x;i+=y;}
        j+=1;
    }re[address(re,p)]=r;im[address(im,p)]=i;
}
#[ruda(launch)]
fn reduction<F:Float>(ar:&kernel_dsl::prelude::Tensor<F>,ai:&kernel_dsl::prelude::Tensor<F>,re:&mut kernel_dsl::prelude::Tensor<F>,im:&mut kernel_dsl::prelude::Tensor<F>,#[comptime] operation:u32) {
    let p=ABSOLUTE_POS as usize;if p<re.len() {
        let mut remaining=p;let mut base=0usize;let mut count=1usize;let mut axis=re.rank();
        while axis>0 {axis-=1;let coordinate=remaining%re.shape(axis);remaining/=re.shape(axis);if re.shape(axis)==ar.shape(axis) {base+=coordinate*ar.stride(axis);}else {count*=ar.shape(axis);}}
        let mut r=F::new(0.0);let mut i=F::new(0.0);if comptime!(operation==98) {r=F::new(1.0);}
        for j in 0..count {let mut rem=j;let mut index=base;let mut d=re.rank();while d>0 {d-=1;if re.shape(d)!=ar.shape(d) {index+=(rem%ar.shape(d))*ar.stride(d);rem/=ar.shape(d);}}
            if comptime!(operation==98) {let rr=r*ar[index]-i*ai[index];let ii=r*ai[index]+i*ar[index];r=rr;i=ii;}
            else {r+=ar[index];i+=ai[index];}}
        if comptime!(operation==107) {r/=F::cast_from(count);i/=F::cast_from(count);}
        re[address(re,p)]=r;im[address(im,p)]=i;
    }
}
#[ruda(launch)]
fn triangular_bytes(a:&kernel_dsl::prelude::Tensor<u8>,out:&mut kernel_dsl::prelude::Tensor<u8>,diagonal:i64,#[comptime] upper:bool) {
    let p=ABSOLUTE_POS as usize;if p<out.len() {
        let rank=out.rank();let width=out.shape(rank-1);let columns=out.shape(rank-2);let rows=out.shape(rank-3);
        let element=p/width;let column=element%columns;let row=element/columns%rows;
        let difference=i64::cast_from(column)-i64::cast_from(row);let mut retain=difference<=diagonal;
        if comptime!(upper) {retain=difference>=diagonal;}
        let mut value=0u8;if retain {value=a[address(a,p)];}out[address(out,p)]=value;
    }
}
