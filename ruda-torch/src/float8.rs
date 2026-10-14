use super::*;
use ruda_kernel::quantization::float8 as kernels;

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_float8_api_version()->u32 {1}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_float8(op:u32,descriptors:*const Descriptor,count:usize,
    transpose_a:bool,transpose_b:bool,strategy:u32)->i32 {
    checked(|| {
        assert!(op<=1 && strategy<=2);
        assert_eq!(count,if op==0 {5} else {3});
        let ds=unsafe {abi::slice(descriptors,count)};
        let vs:Vec<_>=ds.iter().map(|d|unsafe {abi::view(d)}).collect();
        for v in &vs {
            let mut stride=1usize;
            for i in (0..v.shape.len()).rev() {
                assert!(v.shape[i]<=1 || v.strides[i]==stride);
                stride=stride.checked_mul(v.shape[i]).expect("FP8 shape overflow");
            }
            assert!(v.len<=u32::MAX as usize);
        }
        let c=client();
        if op==0 {
            assert!(matches!(vs[0].dtype,12|13) && matches!(vs[1].dtype,12|13));
            assert_eq!(vs[2].dtype,0);assert_eq!(vs[3].dtype,0);assert_eq!(vs[4].dtype,0);
            let result=kernels::scaled_mm(&primitives::tensor(&vs[0]),&primitives::tensor(&vs[1]),
                &primitives::tensor(&vs[2]),&primitives::tensor(&vs[3]),
                if vs[0].dtype==12 {18} else {19},if vs[1].dtype==12 {18} else {19},
                transpose_a,transpose_b,strategy);
            assert_eq!(result.meta.shape().as_slice(),vs[4].shape.as_slice());
            primitives::store(result,&vs[4]);
        } else {
            assert!(vs[0].dtype<=2 && matches!(vs[2].dtype,12|13));
            assert_eq!(vs[1].dtype,0);assert_eq!(vs[1].len,1);assert_eq!(vs[0].shape,vs[2].shape);
            if vs[0].len!=0 {
                let x=primitives::tensor(&vs[0]);
                unsafe {kernels::quantize::launch::<CudaRuntime>(&c,
                    RudaCount::Static(u32::try_from(vs[0].len.div_ceil(128)).unwrap(),1,1),RudaDim::new_1d(128),
                    x.clone().into_array_arg(),primitives::tensor(&vs[1]).into_array_arg(),primitives::tensor(&vs[2]).into_array_arg(),
                    if vs[2].dtype==12 {18} else {19},x.dtype.into());}
            }
        }
        LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);
    })
}
