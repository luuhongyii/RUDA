use super::*;
use ruda_kernel::quantization::float8 as kernels;

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_float8_api_version()->u32 {1}

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_float8_block_api_version()->u32 {1}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_float8_block(op:u32,descriptors:*const Descriptor,count:usize,
    rows_a:u32,rows_w:u32,block_k:u32,margin:u32,power_of_two:bool,strategy:u32)->i32 {
    checked(|| {
        use ruda_kernel::quantization::float8_block as block;
        assert!(op<=1 && strategy<=2 && margin<=16 && rows_a>0 && rows_w>0 && block_k>0 && block_k%32==0);
        assert!(rows_a.checked_mul(block_k).is_some_and(|n|n<=u32::MAX-32)
            && rows_w.checked_mul(block_k).is_some_and(|n|n<=u32::MAX-32));
        assert_eq!(count,if op==0 {5} else {3});
        let ds=unsafe {abi::slice(descriptors,count)};
        let vs:Vec<_>=ds.iter().map(|d|unsafe {abi::view(d)}).collect();
        for v in &vs {
            let mut stride=1usize;
            for i in (0..v.shape.len()).rev() {
                assert!(v.shape[i]<=1 || v.strides[i]==stride);
                stride=stride.checked_mul(v.shape[i]).expect("FP8 block shape overflow");
            }
            assert!(v.len<=u32::MAX as usize && v.shape.iter().all(|&d|d<=u32::MAX as usize));
        }
        let c=client();
        let format=|dtype|if dtype==12 {18u32} else {19u32};
        if op==0 {
            assert!(matches!(vs[0].dtype,12|13) && matches!(vs[1].dtype,12|13));
            assert_eq!(vs[2].dtype,0);assert_eq!(vs[3].dtype,0);assert_eq!(vs[4].dtype,0);
            let result=block::scaled_mm(&primitives::tensor(&vs[0]),&primitives::tensor(&vs[1]),
                &primitives::tensor(&vs[2]),&primitives::tensor(&vs[3]),rows_a as usize,rows_w as usize,block_k as usize,
                format(vs[0].dtype),format(vs[1].dtype),strategy);
            assert_eq!(result.meta.shape().as_slice(),vs[4].shape.as_slice());
            primitives::store(result,&vs[4]);
        } else {
            assert!(vs[0].dtype<=2 && matches!(vs[1].dtype,12|13));
            assert_eq!(vs[2].dtype,0);assert_eq!(vs[0].shape,vs[1].shape);
            let [rows,cols]:[usize;2]=vs[0].shape.as_slice().try_into().expect("FP8 encoding matrix");
            let groups=cols.div_ceil(block_k as usize);
            assert_eq!(vs[2].shape.as_slice(),&[rows.div_ceil(rows_a as usize),groups]);
            if vs[0].len!=0 {
                let blocks=vs[2].len;
                assert!(blocks<=c.properties().hardware.max_ruda_count.0 as usize);
                let x=primitives::tensor(&vs[0]);
                let maximum=if vs[1].dtype==12 {448.0f32} else {57344.0f32};
                unsafe {block::quantize_blocks::launch::<CudaRuntime>(&c,RudaCount::Static(blocks as u32,1,1),RudaDim::new_1d(32),
                    x.clone().into_array_arg(),primitives::tensor(&vs[1]).into_array_arg(),primitives::tensor(&vs[2]).into_array_arg(),
                    rows as u32,cols as u32,groups as u32,rows_a,block_k,maximum*2.0f32.powi(-(margin as i32)),
                    format(vs[1].dtype),power_of_two,x.dtype.into());}
            }
        }
        LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);
    })
}

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
