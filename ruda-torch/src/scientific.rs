use super::*;
use ruda_core::tensor::{DType, Metadata};
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_contiguous_dtype};

pub(super) type Tensor = RudaTensor<CudaRuntime>;

fn storage_dtype(code: u32) -> DType {
    match code {
        0 | 10 => DType::F32, 1 | 14 => DType::F16, 2 => DType::BF16,
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
pub(super) fn component(v: &View, imaginary: bool) -> View {
    let code = match v.dtype {10=>0,11=>9,14=>1,_=>panic!("complex storage descriptor required")};
    let width = element_bytes(code);
    let result = View { handle: v.handle.clone().offset_start(u64::from(imaginary) * width as u64),
        shape: v.shape.clone(), strides: v.strides.iter().map(|s| s.checked_mul(2).expect("complex stride overflow")).collect(),
        len: v.len, dtype: code };
    if v.dtype==14 {indexable(&result);}
    result
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
    assert!(matches!(out.dtype,10|11|14));
    assert_eq!(re.dtype, im.dtype);
    assert_eq!(re.meta.num_elements(), out.len); assert_eq!(im.meta.num_elements(), out.len);
    if out.dtype==14 {
        real_convert(&tensor_view(re), &component(out, false));
        real_convert(&tensor_view(im), &component(out, true));
    } else {
        assert_eq!(re.dtype, storage_dtype(out.dtype));
        copy_storage(&tensor_view(re), &component(out, false));
        copy_storage(&tensor_view(im), &component(out, true));
    }
}
fn grid(n: usize) -> RudaCount {
    assert!(n<=u32::MAX as usize,"scientific GPU work exceeds U32 indexing");
    RudaCount::Static(u32::try_from(n.div_ceil(128)).expect("scientific grid overflow"), 1, 1)
}
pub(super) fn indexable(v:&View) {
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
    let contiguous = [a,out].iter().all(|view| {
        let mut span=1usize;
        for (&size,&stride) in view.shape.iter().zip(&view.strides).rev() {
            if size>1 && stride!=span {return false;}
            span=span.checked_mul(size).expect("conversion stride overflow");
        }
        true
    });
    macro_rules! run { ($i:ty, $o:ty) => {
        unsafe { cast_storage::launch::<$i, $o, CudaRuntime>(&client(), grid(out.len), RudaDim::new_1d(128),
            a.arg(), out.arg(), out.dtype == 3, contiguous); }
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
    if a.dtype == 16 || out.dtype == 16 {crate::quantized::copy_fp4(a,out);return;}
    if matches!(a.dtype,12|13|15) || matches!(out.dtype,12|13|15) {crate::quantized::convert(a,out);return;}
    if a.dtype >= 10 {
        let re = component(a, false);
        if out.dtype >= 10 {
            real_convert(&re, &component(out, false));
            real_convert(&component(a, true), &component(out, true));
        } else if out.dtype == 3 {
            let im = component(a, true);
            unsafe {
                if a.dtype == 10 { complex_bool::launch::<f32, CudaRuntime>(&client(), grid(out.len), RudaDim::new_1d(128), re.arg(), im.arg(), out.arg()); }
                else if a.dtype == 11 { complex_bool::launch::<f64, CudaRuntime>(&client(), grid(out.len), RudaDim::new_1d(128), re.arg(), im.arg(), out.arg()); }
                else { complex_bool::launch::<f16, CudaRuntime>(&client(), grid(out.len), RudaDim::new_1d(128), re.arg(), im.arg(), out.arg()); }
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
    if matches!(op,206|208) && a.dtype==14 && out.dtype==14 {
        half_sign_copy(a,out,op==206,op==208);return;
    }
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
    assert!(matches!(op,1|2|3|4|8|9|10|11|12|13|14|17|19|20|21|22|23|24|25|26|27|28|29|35|36|37|38|39|40|49|50|57|78|79|80|81|82|83|116|117|118|121|122|123|124|206|208|214|216|217|218|219|220|221|222|223|224|225|226|227|231|232),"scientific operation is not implemented");
    let code = if a.dtype == 9 || a.dtype == 11 || b.dtype == 9 || b.dtype == 11 || out.dtype == 9 || out.dtype == 11 { 9 } else { 0 };
    let is_complex = a.dtype >= 10 || b.dtype >= 10 || out.dtype >= 10;
    assert!(!is_complex || matches!(op, 1 | 2 | 8 | 9 | 10 | 11 | 12 | 17 | 19 | 20 | 21 | 27 | 28 | 29 | 35 | 36 | 37 | 38 | 39 | 40 | 49 | 50 | 78 | 79 | 116 | 117 | 118 | 121 | 206 | 207 | 208 | 214 | 217 | 218 | 219 | 220), "operation is not defined for complex inputs");
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
            if code == 9 { pointwise_transcendental::launch::<f64, CudaRuntime>(&client(), grid(out.len), RudaDim::new_1d(128), ar.into_tensor_arg(), ai.into_tensor_arg(), br.into_tensor_arg(), bi.into_tensor_arg(), oldr.into_tensor_arg(), oldi.into_tensor_arg(), re.clone().into_tensor_arg(), im.clone().into_tensor_arg(), scalar, op, is_complex); }
            else { pointwise_transcendental::launch::<f32, CudaRuntime>(&client(), grid(out.len), RudaDim::new_1d(128), ar.into_tensor_arg(), ai.into_tensor_arg(), br.into_tensor_arg(), bi.into_tensor_arg(), oldr.into_tensor_arg(), oldi.into_tensor_arg(), re.clone().into_tensor_arg(), im.clone().into_tensor_arg(), scalar as f32, op, is_complex); }
        }
        if out.dtype >= 10 { copy_complex(&re, &im, out); }
        else { real_convert(&tensor_view(&re), out); }
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
}

fn matmul(a:&View,b:&View,out:&View) {
    use rublas::tensor_matmul::{gemm,GemmOptions,MatmulStrategy};
    let multiply=|a:Tensor,b:Tensor,c:Option<Tensor>,beta:f64| gemm(a,b,c,GemmOptions{beta,..Default::default()},MatmulStrategy::Ruda).expect("scientific GEMM failed");
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

#[ruda(launch)]
fn gamma_pointwise<F:Float+RudaElement>(input:&kernel_dsl::prelude::Tensor<F>,out:&mut kernel_dsl::prelude::Tensor<F>,order:i64,#[comptime] op:u32) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let x=input[address(input,p)];let mut result=F::new(0.0);
        if comptime!(op==228) {result=F::cast_from(crate::gamma::lgamma(f64::cast_from(x)));}
        else if comptime!(op==229) {result=crate::gamma::digamma::<F>(x);}
        else {result=crate::gamma::polygamma::<F>(x,order);}
        out[address(out,p)]=result;
    }
}

#[ruda(launch)]
fn bessel_pointwise<F:Float+RudaElement>(input:&kernel_dsl::prelude::Tensor<F>,out:&mut kernel_dsl::prelude::Tensor<F>,#[comptime] op:u32,#[comptime] wide:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {out[address(out,p)]=crate::bessel::evaluate::<F>(input[address(input,p)],op,wide);}
}

#[ruda(launch)]
fn elementary_pointwise<F:Float+RudaElement>(ar:&kernel_dsl::prelude::Tensor<F>,ai:&kernel_dsl::prelude::Tensor<F>,br:&kernel_dsl::prelude::Tensor<F>,bi:&kernel_dsl::prelude::Tensor<F>,re:&mut kernel_dsl::prelude::Tensor<F>,im:&mut kernel_dsl::prelude::Tensor<F>,eps:F,#[comptime] op:u32,#[comptime] complex:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<re.len() {
        let (r,i)=crate::elementary::evaluate::<F>(ar[address(ar,p)],ai[address(ai,p)],br[address(br,p)],bi[address(bi,p)],eps,op,complex);
        re[address(re,p)]=r;im[address(im,p)]=i;
    }
}

#[ruda(launch)]
fn linspace_float<F:Float+RudaElement>(first:&kernel_dsl::prelude::Tensor<F>,last:&kernel_dsl::prelude::Tensor<F>,step:&kernel_dsl::prelude::Tensor<F>,out:&mut kernel_dsl::prelude::Tensor<F>) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let delta=step[0];
        let mut value=first[0]+delta*F::cast_from(p);
        if p>=out.len()/2 {value=last[0]-delta*F::cast_from(out.len()-p-1);}
        out[address(out,p)]=value;
    }
}

#[ruda(launch)]
fn linspace_integer<I:Numeric+RudaElement>(first:&kernel_dsl::prelude::Tensor<I>,last:&kernel_dsl::prelude::Tensor<I>,step:&kernel_dsl::prelude::Tensor<f32>,out:&mut kernel_dsl::prelude::Tensor<I>) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let delta=step[0];
        let mut value=f32::cast_from(first[0])+delta*f32::cast_from(p);
        if p>=out.len()/2 {value=f32::cast_from(last[0])-delta*f32::cast_from(out.len()-p-1);}
        out[address(out,p)]=I::cast_from(value);
    }
}

fn linspace(first:&View,last:&View,step:&View,out:&View) {
    assert_eq!(first.len,1);assert_eq!(last.len,1);assert_eq!(step.len,1);
    assert_eq!(first.dtype,out.dtype);assert_eq!(last.dtype,out.dtype);assert!(out.len>1);
    if out.dtype>=10 {
        linspace(&component(first,false),&component(last,false),&component(step,false),&component(out,false));
        linspace(&component(first,true),&component(last,true),&component(step,true),&component(out,true));
        return;
    }
    macro_rules! float {($f:ty)=>{unsafe {linspace_float::launch::<$f,CudaRuntime>(&client(),grid(out.len),RudaDim::new_1d(128),first.arg(),last.arg(),step.arg(),out.arg());}}}
    macro_rules! integer {($i:ty)=>{unsafe {linspace_integer::launch::<$i,CudaRuntime>(&client(),grid(out.len),RudaDim::new_1d(128),first.arg(),last.arg(),step.arg(),out.arg());}}}
    if matches!(out.dtype,0|1|2|9) {assert_eq!(step.dtype,out.dtype);} else {assert_eq!(step.dtype,0);}
    match out.dtype {
        0=>float!(f32),1=>float!(f16),2=>float!(bf16),9=>float!(f64),
        4=>integer!(i64),5=>integer!(i32),6=>integer!(i16),7=>integer!(i8),8=>integer!(u8),
        _=>panic!("unsupported linspace storage"),
    }
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
        let xs: Vec<View> = unsafe { crate::abi::slice(descriptors,count) }.iter().map(|d| unsafe { crate::abi::view(d) }).collect();
        for x in &xs {indexable(x);}
        let ints = if integer_count == 0 { &[][..] } else { unsafe { crate::abi::slice(integers,integer_count) } };
        let values = if scalar_count == 0 { &[][..] } else { unsafe { crate::abi::slice(scalars,scalar_count) } };
        match op {
            656 => {crate::lora_epilogue::projection(&xs,ints,values);}
            654..=655 => {crate::lora_epilogue::dispatch(op,&xs,ints,values);}
            652..=653 => {crate::rotary::dispatch(op,&xs,ints);}
            650..=651 => {crate::loss::dispatch_fused_cross_entropy(op,&xs,ints);}
            579..=583 => {crate::quantized_embedding::dispatch(op,&xs,ints,values);}
            578 => {crate::quantized_assignment::dispatch(&xs,ints,values);}
            576..=577 => {crate::quantized_index::dispatch(op,&xs,ints,values);}
            574..=575 => {crate::quantized_reduce::dispatch(op,&xs,ints,values);}
            572..=573 => {crate::quantized_selection::dispatch(op,&xs,ints,values);}
            570..=571 => {crate::quantized_resize::dispatch(op,&xs,ints,values);}
            568..=569 => {crate::quantized_convolution::dispatch(op,&xs,ints,values);}
            565..=567 => {crate::quantized_linear::dispatch(op,&xs,ints,values);}
            562..=564 => {crate::quantized_pooling::dispatch(op,&xs,ints,values);}
            559..=561 => {crate::quantized_normalization::dispatch(op,&xs,ints,values);}
            556..=558 => {crate::quantized_extended::dispatch(op,&xs,ints,values);}
            554..=555 => {crate::quantized_arithmetic::dispatch(op,&xs,ints,values);}
            552..=553 => {crate::quantized_activation::dispatch(op,&xs,ints,values);}
            549..=551 => {crate::affine_quantized::dispatch(op,&xs,ints,values);}
            545..=548 => {crate::fake_quant_affine::dispatch(op,&xs,ints,values);}
            542..=544 => {assert!(values.is_empty());crate::sparse_solve::dispatch(op,&xs,ints);}
            540..=541 => {assert!(ints.is_empty() && values.is_empty());crate::sparse_batched_hybrid::dispatch(op,&xs);}
            538..=539 => {assert!(values.is_empty());crate::sparse_indexing::dispatch(op,&xs,ints);}
            536..=537 => {assert!(values.is_empty());crate::sparse_triangular::dispatch(op,&xs,ints);}
            533..=535 => {assert!(values.is_empty());crate::spmm_reduce::dispatch(op,&xs,ints);}
            530..=532 => {assert!(values.is_empty());crate::sparse_softmax::dispatch(op,&xs,ints);}
            528..=529 => {assert!(values.is_empty());crate::compressed_reduce::dispatch(op,&xs,ints);}
            525..=527 => {assert!(values.is_empty());crate::compressed_add::dispatch(op,&xs,ints);}
            523..=524 => {assert!(ints.is_empty());crate::compressed_spgemm::dispatch(op,&xs,values);}
            522 => {assert!(ints.is_empty());crate::sampled_matmul::dispatch(&xs,values);}
            521 => {assert!(ints.is_empty());crate::compressed_matmul::dispatch(&xs,values);}
            519..=520 => {assert!(values.is_empty());crate::sparse_blocks::dispatch(op,&xs,ints);}
            516..=518 => {assert!(values.is_empty());crate::sparse_compressed::dispatch(op,&xs,ints);}
            515 => {assert!(values.is_empty());crate::sparse_reduce::dispatch(&xs,ints);}
            512..=514 => {assert!(values.is_empty());crate::sparse_product::dispatch(op,&xs,ints);}
            508..=511 => {assert!(ints.is_empty());crate::sparse_algebra::dispatch(op,&xs,values);}
            502..=507 => {assert!(values.is_empty());crate::sparse_coo::dispatch(op,&xs,ints);}
            501 => {assert!(values.is_empty());crate::grouped_matmul::dispatch(&xs,ints);}
            500 => {assert!(values.is_empty());crate::fp4_matmul::dispatch(&xs,ints);}
            499 => {assert!(values.is_empty());crate::scaled_matmul::dispatch(&xs,ints);}
            498 => {assert!(ints.is_empty() && values.is_empty());crate::int_matmul::dispatch(&xs);}
            497 => {assert!(values.is_empty());crate::dot_ops::dispatch(&xs,ints);}
            496 => {crate::blas_accumulate::dispatch(&xs,ints,values);}
            495 => {assert!(values.is_empty());crate::linear_combination::dispatch(&xs,ints);}
            494 => {assert!(values.is_empty());crate::rank_one::dispatch(&xs,ints);}
            493 => {assert!(ints.is_empty());crate::kaiser_window::dispatch(&xs,values);}
            492 => {crate::logarithmic_ops::dispatch(&xs,ints,values);}
            491 => {assert!(values.is_empty());crate::trigonometric_ops::dispatch(&xs,ints);}
            490 => {assert!(values.is_empty());crate::exponential_ops::dispatch_backward(&xs,ints);}
            489 => {assert!(values.is_empty());crate::exponential_ops::dispatch(&xs,ints);}
            488 => {assert!(values.is_empty());crate::unary_basic::dispatch(&xs,ints);}
            487 => {assert!(values.is_empty());crate::power_ops::dispatch(&xs,ints);}
            486 => {assert!(values.is_empty());crate::division_ops::dispatch(&xs,ints);}
            485 => {assert!(values.is_empty());crate::arithmetic_ops::dispatch(&xs,ints);}
            484 => {assert!(values.is_empty());crate::predicate_ops::dispatch(&xs,ints);}
            482..=483 => {assert!(values.is_empty());crate::conditional_select::dispatch(op,&xs,ints);}
            481 => {assert!(values.is_empty());crate::elementwise_extrema::dispatch(&xs,ints);}
            480 => {assert!(values.is_empty());crate::additive_reduce::dispatch(&xs,ints);}
            479 => {assert!(values.is_empty());crate::product_reduce::dispatch(&xs,ints);}
            478 => {assert!(values.is_empty());crate::logical_reduce::dispatch(&xs,ints);}
            476..=477 => {assert!(values.is_empty());crate::indexed_write::dispatch(op,&xs,ints);}
            475 => {assert!(values.is_empty());crate::advanced_index::dispatch(&xs,ints);}
            472..=474 => {assert!(values.is_empty());crate::indexed_write::dispatch(op,&xs,ints);}
            470..=471 => {assert!(values.is_empty());crate::indexed_data::dispatch(op,&xs,ints);}
            468..=469 => {assert!(values.is_empty());crate::embedding_ops::dispatch(op,&xs,ints);}
            467 => {assert!(values.is_empty());crate::scatter_reduce_ops::dispatch(&xs,ints);}
            463..=466 => {assert!(values.is_empty());crate::mask_compact::dispatch(op,&xs,ints);}
            461..=462 => {assert!(values.is_empty());crate::unique_ops::dispatch(op,&xs,ints);}
            460 => {crate::variance_reduce::dispatch(&xs,ints,values);}
            457..=459 => {assert!(values.is_empty());crate::nan_log_reduce::dispatch(op,&xs,ints);}
            455..=456 => {crate::vector_norm::dispatch(op,&xs,ints,values);}
            454 => {assert!(values.is_empty());crate::foreach_reduce::dispatch(&xs,ints);}
            453 => {assert!(values.is_empty());crate::foreach_ternary::dispatch(&xs,ints);}
            452 => {assert!(values.is_empty());crate::foreach_binary::dispatch(&xs,ints);}
            451 => {assert!(values.is_empty());crate::foreach_copy::dispatch(&xs,ints);}
            449..=450 => {assert!(values.is_empty());crate::foreach_unary::dispatch(op,&xs,ints);}
            447..=448 => {assert!(values.is_empty());crate::weight_norm::dispatch(op,&xs,ints);}
            443..=446 => {assert!(values.is_empty());crate::rnn_cell::dispatch(op,&xs,ints);}
            440..=442 => {crate::random_activation::dispatch(op,&xs,ints,values);}
            436..=439 => {crate::fused_optim_ops::dispatch(op,&xs,ints,values);}
            434..=435 => {crate::amp_ops::dispatch(op,&xs,ints,values);}
            430..=433 => {crate::random_base::dispatch(op,&xs,ints,values);}
            428..=429 => {assert!(ints.is_empty() && values.is_empty());crate::distribution_grad::dispatch(op,&xs);}
            425..=427 => {assert!(values.is_empty());crate::distribution_ops::dispatch(op,&xs,ints);}
            424 => {assert!(values.is_empty());crate::poisson_ops::dispatch(&xs,ints);}
            422..=423 => {assert!(values.is_empty());crate::randperm_ops::dispatch(op,&xs,ints);}
            419..=421 => {assert!(values.is_empty());crate::multinomial_ops::dispatch(op,&xs,ints);}
            414..=418 => {crate::random_extra::dispatch(op,&xs,ints,values);}
            413 => {assert!(ints.is_empty() && values.is_empty());crate::logspace::dispatch(&xs);}
            411..=412 => {assert!(values.is_empty());crate::triangular_indices::dispatch(op,&xs,ints);}
            407..=409 => {assert!(ints.is_empty() && values.is_empty());crate::activation_extra::dispatch(op,&xs);}
            405..=406 => {assert!(values.is_empty());crate::cumulative_ops::dispatch(op,&xs,ints);}
            404 => {assert!(values.is_empty());crate::rearrange_ops::roll_dispatch(&xs,ints);}
            402..=403 => {assert!(values.is_empty());crate::membership_ops::dispatch(op,&xs,ints);}
            399..=401 => {assert!(values.is_empty());crate::ternary_ops::dispatch(op,&xs,ints);}
            397..=398 => {assert!(ints.is_empty() && values.is_empty());crate::bitwise_ops::dispatch(op,&xs);}
            396 => {crate::complex32_ops::dispatch(&xs,ints,values);}
            389..=395 => {crate::complex_classify::dispatch(op,&xs,ints,values);}
            384..=388 => {assert!(values.is_empty());crate::math_extra::dispatch(op,&xs,ints);}
            380..=383 => {assert!(ints.is_empty() && values.is_empty());crate::gamma_extra::dispatch(op,&xs);}
            368..=379 => {assert!(ints.is_empty() && values.is_empty());crate::polynomial_ops::dispatch(op,&xs);}
            365 => {assert!(values.is_empty());crate::log_scan_ops::dispatch(&xs,ints);}
            366..=367 => {crate::segment_ops::dispatch(op,&xs,ints,values);}
            361..=364 => {crate::margin_ops::dispatch(op,&xs,ints,values);}
            356..=360 => {assert!(values.is_empty());crate::ctc_ops::dispatch(op,&xs,ints);}
            354..=355 => {assert!(values.is_empty());crate::selection_ops::dispatch(op,&xs,ints);}
            350..=353 => {crate::histogram_ops::dispatch(op,&xs,ints,values);}
            345..=349 => {crate::indexed_ops::dispatch(op,&xs,ints,values);}
            341..=344 => {assert!(values.is_empty());crate::embedding_bag_ops::dispatch(op,&xs,ints);}
            339..=340 => {crate::distance_ops::dispatch(op,&xs,ints,values);}
            335..=338 => {assert!(values.is_empty());crate::rearrange_ops::dispatch(op,&xs,ints);}
            333..=334 => {crate::lanczos_ops::dispatch(op,&xs,ints,values);}
            331..=332 => {assert!(values.is_empty());crate::grid_ops::dispatch(op,&xs,ints);}
            325..=330 => {crate::resize_ops::dispatch(op,&xs,ints,values);}
            323..=324 => {assert!(values.is_empty());crate::padding_ops::dispatch(op,&xs,ints);}
            320..=322 => {assert!(values.is_empty());crate::pooling_ops::dispatch_indexed(op,&xs,ints);}
            318..=319 => {assert!(values.is_empty());crate::pooling_ops::dispatch(op,&xs,ints);}
            309..=317 => {crate::normalization_ops::dispatch(op,&xs,ints,values);}
            293..=304 => {crate::loss::dispatch(op,&xs,ints,values);}
            305..=308 => {
                assert_eq!(xs.len(),3);assert_eq!(ints.len(),1);
                crate::softmax::wide_launch(op,&xs[0],&xs[1],&xs[2],usize::try_from(ints[0]).expect("negative softmax dimension"));
            }
            265..=292 => {crate::activation::dispatch(op,&xs,ints,values);}
            258..=264 => {assert_eq!(count,if op==262 || op==263 {2}else{3});crate::floating::dispatch(op,&xs,ints);}
            249..=257 => {
                assert_eq!(count,3);assert_eq!(values.len(),1);
                let a=&xs[0];let b=&xs[1];let out=&xs[2];
                assert_eq!(a.shape,b.shape);assert_eq!(a.shape,out.shape);
                assert_eq!(a.dtype,b.dtype);assert_eq!(a.dtype,out.dtype);
                let complex=out.dtype>=10;assert!(!complex || op<=253);
                let code=if out.dtype==9 || out.dtype==11 {9}else{0};
                let (ar,ai)=parts_as(a,code);let(br,bi)=parts_as(b,code);
                let re=allocate(&out.shape,storage_dtype(code));let im=allocate(&out.shape,storage_dtype(code));
                if out.len!=0 {unsafe {
                    if code==9 {elementary_pointwise::launch::<f64,CudaRuntime>(&client(),grid(out.len),RudaDim::new_1d(128),ar.into_tensor_arg(),ai.into_tensor_arg(),br.into_tensor_arg(),bi.into_tensor_arg(),re.clone().into_tensor_arg(),im.clone().into_tensor_arg(),values[0],op,complex);}
                    else {elementary_pointwise::launch::<f32,CudaRuntime>(&client(),grid(out.len),RudaDim::new_1d(128),ar.into_tensor_arg(),ai.into_tensor_arg(),br.into_tensor_arg(),bi.into_tensor_arg(),re.clone().into_tensor_arg(),im.clone().into_tensor_arg(),values[0] as f32,op,complex);}
                }}
                if complex {copy_complex(&re,&im,out);}else {real_convert(&tensor_view(&re),out);}
            }
            248 => {assert_eq!(count,4);linspace(&xs[0],&xs[1],&xs[2],&xs[3]);}
            228 | 229 | 230 | 233..=247 => {
                assert_eq!(count,2);assert_eq!(ints.len(),1);assert!(ints[0]>=0);
                let a=&xs[0];let out=&xs[1];assert!(a.dtype<10 && out.dtype<10);
                assert_eq!(a.shape,out.shape);assert_eq!(a.dtype,out.dtype);
                let code=if a.dtype==9 {9}else{0};let input=parts_as(a,code).0;
                let output=allocate(&out.shape,storage_dtype(code));
                if out.len!=0 {unsafe {
                    if op>=233 {
                        if code==9 {bessel_pointwise::launch::<f64,CudaRuntime>(&client(),grid(out.len),RudaDim::new_1d(128),input.into_tensor_arg(),output.clone().into_tensor_arg(),op,true);}
                        else {bessel_pointwise::launch::<f32,CudaRuntime>(&client(),grid(out.len),RudaDim::new_1d(128),input.into_tensor_arg(),output.clone().into_tensor_arg(),op,false);}
                    } else if code==9 {gamma_pointwise::launch::<f64,CudaRuntime>(&client(),grid(out.len),RudaDim::new_1d(128),input.into_tensor_arg(),output.clone().into_tensor_arg(),ints[0],op);}
                    else {gamma_pointwise::launch::<f32,CudaRuntime>(&client(),grid(out.len),RudaDim::new_1d(128),input.into_tensor_arg(),output.clone().into_tensor_arg(),ints[0],op);}
                }}
                real_convert(&tensor_view(&output),out);
            }
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
                if xs[0].dtype==16 {
                    assert!(ints[0]==0 && ints[1]==0,"packed FP4 logical sign views are not supported");
                    crate::quantized::copy_fp4(&xs[0],&xs[1]);
                }else if xs[0].dtype==14 {
                    half_sign_copy(&xs[0],&xs[1],ints[0]!=0,ints[1]!=0);
                }else if matches!(xs[0].dtype,12|13|15) {
                    assert_eq!(ints[0],0,"FP8 conjugate views are not defined");
                    crate::quantized::resolve(&xs[0],&xs[1],ints[1]!=0);
                }else if (3..=8).contains(&xs[0].dtype) {
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

#[ruda(launch)]
fn half_sign_kernel(ar:&kernel_dsl::prelude::Tensor<u16>,ai:&kernel_dsl::prelude::Tensor<u16>,out_r:&mut kernel_dsl::prelude::Tensor<u16>,out_i:&mut kernel_dsl::prelude::Tensor<u16>,#[comptime] conjugate:bool,#[comptime] negative:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out_r.len() {
        let mut re=ar[address(ar,p)];let mut im=ai[address(ai,p)];
        if comptime!(negative) {re^=0x8000u16;}
        if comptime!(conjugate!=negative) {im^=0x8000u16;}
        out_r[address(out_r,p)]=re;out_i[address(out_i,p)]=im;
    }
}

fn half_sign_copy(a:&View,out:&View,conjugate:bool,negative:bool) {
    assert_eq!((a.dtype,out.dtype),(14,14));assert_eq!(a.shape,out.shape);
    if out.len==0 {return;}
    unsafe {half_sign_kernel::launch::<CudaRuntime>(&client(),grid(out.len),RudaDim::new_1d(128),
        component(a,false).arg(),component(a,true).arg(),component(out,false).arg(),component(out,true).arg(),conjugate,negative);}
    LAUNCHES.fetch_add(1,Ordering::Relaxed);
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
pub(super) fn address<T: Numeric>(x: &kernel_dsl::prelude::Tensor<T>, position: usize) -> usize {
    let mut remaining=position; let mut offset=0usize; let mut axis=x.rank();
    while axis>0 {axis-=1;offset+=(remaining%x.shape(axis))*x.stride(axis);remaining/=x.shape(axis);}
    offset
}
#[ruda(launch)]
fn cast_storage<I: Numeric+RudaElement,O: Numeric+RudaElement>(a:&kernel_dsl::prelude::Tensor<I>,out:&mut kernel_dsl::prelude::Tensor<O>,#[comptime] boolean:bool,#[comptime] contiguous:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let mut ai=p;let mut oi=p;
        if !comptime!(contiguous) {ai=address(a,p);oi=address(out,p);}
        let x=a[ai];let mut value=O::cast_from(x);if comptime!(boolean) {value=O::cast_from(x!=I::cast_from(0u32));}out[oi]=value;
    }
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
fn scale<F:Float+RudaElement>(a:&kernel_dsl::prelude::Tensor<F>,out:&mut kernel_dsl::prelude::Tensor<F>,factor:F) {
    let p=ABSOLUTE_POS as usize;if p<out.len() {out[address(out,p)]=a[address(a,p)]*factor;}
}
#[ruda(launch)]
fn constant<F:Float+RudaElement>(re:&mut kernel_dsl::prelude::Tensor<F>,im:&mut kernel_dsl::prelude::Tensor<F>,r:F,i:F) {
    let p=ABSOLUTE_POS as usize;if p<re.len() {re[address(re,p)]=r;im[address(im,p)]=i;}
}
#[ruda]
fn signed<F:Float>(magnitude:F,sign:F)->F {
    let mut result=magnitude.abs();
    if sign<F::new(0.0) || (sign==F::new(0.0) && F::new(1.0)/sign<F::new(0.0)) {result=-result;}
    result
}
#[ruda]
pub(super) fn complex_log<F:Float>(x:F,y:F)->(F,F) {
    let mut scale=x.abs();if y.abs()>scale {scale=y.abs();}
    let mut real=scale.ln();
    if scale!=F::new(0.0) && scale!=F::new(f32::INFINITY) {
        let a=x/scale;let b=y/scale;real+=(a*a+b*b).ln()/F::new(2.0);
    }
    if x.abs()==F::new(f32::INFINITY) || y.abs()==F::new(f32::INFINITY) {real=F::new(f32::INFINITY);}
    else if x!=x || y!=y {real=F::new(f32::NAN);}
    (real,y.atan2(x))
}
#[ruda]
pub(super) fn complex_sqrt<F:Float>(x:F,y:F)->(F,F) {
    let z=F::new(0.0f32);let two=F::new(2.0f32);let inf=F::new(f32::INFINITY);
    let mut r=F::new(f32::NAN);let mut i=F::new(f32::NAN);
    if y.abs()==inf {r=inf;i=y;}
    else if x.abs()==inf {
        if x>z {r=x;i=signed::<F>(z,y);if y!=y {i=y;}}
        else {r=z;i=signed::<F>(inf,y);if y!=y {r=y;}}
    }
    else if x==x && y==y {
        let ax=x.abs();let ay=y.abs();let mut scale=ax;if ay>scale {scale=ay;}
        if scale==z {r=z;i=y;}
        else {
            let magnitude=(x/scale).hypot(y/scale);
            let t=scale.sqrt()*((magnitude+ax/scale)/two).sqrt();
            if x>=z {r=t;i=(y/t)/two;}
            else {r=(ay/t)/two;i=signed::<F>(t,y);}
        }
    }
    (r,i)
}
#[ruda]
pub(super) fn inverse_hyperbolic<F:Float>(x:F,y:F,#[comptime] acosh:bool)->(F,F) {
    let z=F::new(0.0);let one=F::new(1.0);let two=F::new(2.0);let half=F::new(0.5);let inf=F::new(f32::INFINITY);
    let mut a=x.abs();let mut b=y.abs();
    if comptime!(acosh) {a=y.abs();b=x.abs();}
    let mut scale=a;if b>scale {scale=b;}
    let mut r=F::new(f32::NAN);let mut i=r;
    if a==inf || b==inf || scale>F::new(1.0e16) {
        let (lr,_li)=complex_log::<F>(a,b);r=lr+two.ln();
        if a==inf || b==inf {r=inf;}
        if comptime!(acosh) {i=signed::<F>(a.atan2(x),y);}
        else {r=signed::<F>(r,x);i=signed::<F>(b.atan2(a),y);if x!=x && b==inf {r=signed::<F>(inf,y);}}
    } else if a==a && b==b {
        let plus=a.hypot(b+one);let minus=a.hypot(b-one);let average=plus*half+minus*half;
        let mut d=z;
        if a==z {
            if b<=one {r=z;d=((one-b)*(one+b)).sqrt();}
            else {r=b.acosh();}
        } else {
            let q=a/(plus+b+one);let t=a/(minus+(one-b).abs());let sum=q+t;
            if b<=one {
                r=two*(half*a.sqrt()*sum.sqrt()).asinh();
                if a*a==z && b<one {r=a/((one-b)*(one+b)).sqrt();}
            } else {
                let am1=(b-one)+half*a*sum;
                r=(am1+am1.sqrt()*(average+one).sqrt()).log1p();
            }
            if b>=one {d=a.sqrt()*((average+b)*half*sum).sqrt();}
            else {d=((average+b)*((one-b)+half*a*sum)).sqrt();}
        }
        if comptime!(acosh) {i=signed::<F>(d.atan2(x),y);}
        else {r=signed::<F>(r,x);i=signed::<F>(b.atan2(d),y);}
    } else if comptime!(!acosh) {
        if y==z {r=x;i=y;}
    }
    (r,i)
}
#[ruda]
pub(super) fn complex_tanh<F:Float>(x:F,y:F)->(F,F) {
    let z=F::new(0.0);let one=F::new(1.0);let two=F::new(2.0);let inf=F::new(f32::INFINITY);
    let mut r=F::new(f32::NAN);let mut i=r;
    if x.abs()==inf {
        r=signed::<F>(one,x);i=signed::<F>(z,y);
        if y==y && y.abs()!=inf {i=signed::<F>(z,y.sin()*y.cos());}
    }
    else if y==y && y.abs()!=inf && x==x {
        let sine=y.sin();let cosine=y.cos();
        if x.abs()<F::new(20.0) {
            let s=x.sinh();let c=x.cosh();let denominator=s*s+cosine*cosine;
            r=s*c/denominator;i=sine*cosine/denominator;
        } else {
            let t=(-two*x.abs()).exp();let delta=one-t;let denominator=delta*delta+F::new(4.0)*t*cosine*cosine;
            r=signed::<F>(delta*(one+t)/denominator,x);i=F::new(4.0)*t*sine*cosine/denominator;
        }
        if y==z {i=y;}
    } else {
        if y==z {i=y;}
    }
    (r,i)
}
#[ruda]
pub(super) fn complex_atanh<F:Float>(x:F,y:F)->(F,F) {
    let z=F::new(0.0);let one=F::new(1.0);let two=F::new(2.0);let four=F::new(4.0);let inf=F::new(f32::INFINITY);
    let a=x.abs();let b=y.abs();let mut r=F::new(f32::NAN);let mut i=r;
    if a==inf || b==inf {r=signed::<F>(z,x);i=signed::<F>(one.atan2(z),y);if y!=y {i=y;}}
    else if x==x && y==y {
        let mut scale=a;if b>scale {scale=b;}if scale<one {scale=one;}
        let reciprocal=one/scale;let xx=a/scale;let yy=y/scale;let left=(one-a)/scale;
        let denominator=left*left+yy*yy;let ratio=((four*xx)/scale)/denominator;
        r=ratio.log1p()/four;
        if denominator==z || ratio==inf {
            let (lr,_li)=complex_log::<F>(one+a,y);let (rr,_ri)=complex_log::<F>(one-a,y);r=(lr-rr)/two;
        }
        r=signed::<F>(r,x);
        i=((two*yy)/scale).atan2((reciprocal-xx)*(reciprocal+xx)-yy*yy)/two;
        if x==z {r=x;}
    } else if x==z {r=x;}
    (r,i)
}
#[ruda(launch)]
fn pointwise_transcendental<F:Float+RudaElement>(ar:&kernel_dsl::prelude::Tensor<F>,ai:&kernel_dsl::prelude::Tensor<F>,br:&kernel_dsl::prelude::Tensor<F>,bi:&kernel_dsl::prelude::Tensor<F>,oldr:&kernel_dsl::prelude::Tensor<F>,oldi:&kernel_dsl::prelude::Tensor<F>,re:&mut kernel_dsl::prelude::Tensor<F>,im:&mut kernel_dsl::prelude::Tensor<F>,alpha:F,#[comptime] op:u32,#[comptime] complex:bool) {
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
        else if comptime!(op==10) {if comptime!(complex) {let (a,b)=complex_log::<F>(x,u);r=a;i=b;}else {r=x.ln();i=z;}}
        else if comptime!(op==11 || op==12) {
            if comptime!(complex) {let (a,b)=complex_sqrt::<F>(x,u);r=a;i=b;
                if comptime!(op==12) {let rr=r;let ii=i;
                    if ii.abs()>rr.abs() {let ratio=rr/ii;let den=ii+rr*ratio;r=ratio/den;i=-one/den;}
                    else {let ratio=ii/rr;let den=rr+ii*ratio;r=one/den;i=-ratio/den;}
                }
            } else {r=x.sqrt();i=z;if comptime!(op==12) {r=one/r;}}
        }
        else if comptime!(op==13) {r=one/(one+(-x).exp());i=z;}
        else if comptime!(op==14) {r=x/(one+(-x).exp());i=z;}
        else if comptime!(op==17) {r=x.tanh();i=z;if comptime!(complex) {let (a,b)=complex_tanh::<F>(x,u);r=a;i=b;}}
        else if comptime!(op==19) {r=x.sin();i=z;if comptime!(complex) {r*=u.cosh();i=x.cos()*u.sinh();}}
        else if comptime!(op==20) {r=x.cos();i=z;if comptime!(complex) {r*=u.cosh();i=-x.sin()*u.sinh();}}
        else if comptime!(op==21) {r=x.hypot(u);i=z;}
        else if comptime!(op==22) {r=z;if x>z {r=one;}else if x<z {r=-one;}}
        else if comptime!(op==23) {r=x.floor();}
        else if comptime!(op==24) {r=x.ceil();}
        else if comptime!(op==25) {r=x.trunc();}
        else if comptime!(op==26) {r=x.round();}
        else if comptime!(op==35) {r=x.log1p();i=z;if comptime!(complex) {
            let (a,b)=complex_log::<F>(one+x,u);r=a;i=b;
            if x.abs()<F::new(0.5) && u.abs()<F::new(0.5) {r=(F::new(2.0)*x+x*x+u*u).log1p()/F::new(2.0);}
        }}
        else if comptime!(op==36) {r=x.sinh();i=z;if comptime!(complex) {r*=u.cos();i=x.cosh()*u.sin();}}
        else if comptime!(op==37) {r=x.cosh();i=z;if comptime!(complex) {r*=u.cos();i=x.sinh()*u.sin();}}
        else if comptime!(op==38) {r=x.asinh();i=z;if comptime!(complex) {let (a,b)=inverse_hyperbolic::<F>(x,u,false);r=a;i=b;}}
        else if comptime!(op==39) {r=x.acosh();i=z;if comptime!(complex) {let (a,b)=inverse_hyperbolic::<F>(x,u,true);r=a;i=b;}}
        else if comptime!(op==40) {r=x.atanh();i=z;if comptime!(complex) {let (a,b)=complex_atanh::<F>(x,u);r=a;i=b;}}
        else if comptime!(op==217) {r=x.tan();i=z;if comptime!(complex) {let (a,b)=complex_tanh::<F>(-u,x);r=b;i=-a;}}
        else if comptime!(op==218) {r=x.asin();i=z;if comptime!(complex) {let (a,b)=inverse_hyperbolic::<F>(-u,x,false);r=b;i=-a;if u!=u && x.abs()==F::new(f32::INFINITY) {i=x;}}}
        else if comptime!(op==219) {r=x.acos();i=z;if comptime!(complex) {let (a,b)=inverse_hyperbolic::<F>(x,u,true);r=b.abs();i=-signed::<F>(a,u);if x==z && u!=u {r=one.atan2(z);}}}
        else if comptime!(op==220) {r=x.atan();i=z;if comptime!(complex) {let (a,b)=complex_atanh::<F>(-u,x);r=b;i=-a;}}
        else if comptime!(op==221) {r=x.atan2(y);i=z;}
        else if comptime!(op==222) {r=x.hypot(y);i=z;}
        else if comptime!(op==223) {r=crate::special::erfc::<F>(x,false);i=z;}
        else if comptime!(op==224) {r=crate::special::erfc::<F>(x,true);i=z;}
        else if comptime!(op==225) {r=crate::special::erfinv::<F>(x);i=z;}
        else if comptime!(op==226) {r=crate::special::log_ndtr::<F>(x);i=z;}
        else if comptime!(op==227) {r=crate::special::ndtri::<F>(x);i=z;}
        else if comptime!(op==231 || op==232) {
            if y!=y {r=y;}
            else if x==z {r=z;}
            else if comptime!(op==231) {r=x*y.ln();}
            else {r=x*y.log1p();}
            i=z;
        }
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
