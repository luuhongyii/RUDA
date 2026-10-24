//! Descriptor bridge to ruDNN's dense online SDPA kernels.
use super::*;
use rudnn::dense_attention as kernel;
use ruda_core::tensor::DType;

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_dense_attention_api_version() -> u32 { 1 }

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_indexed_dense_attention_api_version() -> u32 { 1 }

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_attention_dropout_api_version() -> u32 { 1 }

pub(super) fn validate_dropout(vs: &[View], ints: &[i64], p: f32) -> u64 {
    assert_eq!(vs.len(),2);assert_eq!(ints.len(),7);assert!((0.0..=1.0).contains(&p));
    let output=&vs[1];packed(output);assert!(output.dtype<=2);assert_eq!(vs[0].dtype,output.dtype);
    assert_eq!(vs[0].len,0);assert_eq!(output.shape.len(),4);
    assert_eq!(output.shape[0] as i64,ints[0]);assert!(ints[0]>=0);
    let mut scores=ints[0];
    for axis in 1..4 {
        assert!(ints[axis]>=0 && ints[axis+3]>=0 && (output.shape[axis] as i64)<=ints[axis]
            && ints[axis+3]<=ints[axis]-output.shape[axis] as i64);
        scores=scores.checked_mul(ints[axis]).expect("dropout logical size overflow");
    }
    (scores as u64).div_ceil(4)
}

pub(super) fn prepare_dropout(c: &ComputeClient<CudaRuntime>, vs: &[View], ints: &[i64],
    p: f32, seed: u64, counter: u64) -> PreparedKernel<CudaRuntime> {
    counter.checked_add(validate_dropout(vs,ints,p)).expect("dropout RNG counter exhausted");
    let out=&vs[1];let tensor=primitives::tensor(out);
    unsafe { kernel::dropout_mask::prepare::<CudaRuntime>(c,
        RudaCount::Static(out.len.div_ceil(128).max(1) as u32,1,1),RudaDim::new_1d(128),tensor.clone().into_array_arg(),
        out.shape[1] as u32,out.shape[2] as u32,out.shape[3] as u32,
        ints[1],ints[2],ints[3],ints[4],ints[5],ints[6],p,
        seed as u32,(seed>>32) as u32,counter as u32,(counter>>32) as u32,tensor.dtype.into()) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_attention_dropout(descriptors: *const Descriptor,
    integers: *const i64, p: f32, seed: u64, counter: u64) -> i32 {
    checked(|| {
        let vs:Vec<_>=unsafe { crate::abi::slice(descriptors,2) }.iter().map(|d|unsafe { crate::abi::view(d) }).collect();
        let ints=unsafe { crate::abi::slice(integers,7) };
        counter.checked_add(validate_dropout(&vs,ints,p)).expect("dropout RNG counter exhausted");
        let c=client();let out=&vs[1];let tensor=primitives::tensor(out);
        if out.len>0 {
            unsafe { kernel::dropout_mask::launch::<CudaRuntime>(&c,
                RudaCount::Static(out.len.div_ceil(128) as u32,1,1),RudaDim::new_1d(128),tensor.clone().into_array_arg(),
                out.shape[1] as u32,out.shape[2] as u32,out.shape[3] as u32,
                ints[1],ints[2],ints[3],ints[4],ints[5],ints[6],p,
                seed as u32,(seed>>32) as u32,counter as u32,(counter>>32) as u32,tensor.dtype.into()); }
            LAUNCHES.fetch_add(1,Ordering::Relaxed);
        }
        finish_dispatch(&c);
    })
}

fn packed(view: &View) {
    let mut stride = 1usize;
    for d in (0..view.shape.len()).rev() {
        assert!(view.shape[d] <= 1 || view.strides[d] == stride, "dense attention requires contiguous storage");
        stride = stride.checked_mul(view.shape[d]).expect("dense attention shape overflow");
    }
    assert!(view.len <= u32::MAX as usize, "dense attention exceeds U32 indexing");
}

/// Forward descriptors: Q,K,V,expanded-mask,O,statistics.
/// Backward descriptors: Q,K,V,expanded-mask,dO,statistics,dQ,dK,dV,dMask.
/// The adapter owns all descriptors and disjoint outputs for the entire call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_dense_attention(op: u32, descriptors: *const Descriptor,
    count: usize, mask_kind: u32, causal: bool, scale: f32, dropout_p: f32,
    seed: u64, counter: u64, need_mask: bool) -> i32 {
    unsafe { dispatch(op, descriptors, count, mask_kind, causal, scale, dropout_p, seed, counter, need_mask, None) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_indexed_dense_attention(op: u32, descriptors: *const Descriptor,
    count: usize, mask_kind: u32, causal: bool, scale: f32, dropout_p: f32,
    seed: u64, counter: u64, need_mask: bool, global_heads: i64, head_start: i64) -> i32 {
    unsafe { dispatch(op, descriptors, count, mask_kind, causal, scale, dropout_p, seed, counter, need_mask,
        Some((global_heads, head_start))) }
}

unsafe fn dispatch(op: u32, descriptors: *const Descriptor, count: usize, mask_kind: u32,
    causal: bool, scale: f32, dropout_p: f32, seed: u64, counter: u64, need_mask: bool,
    head_partition: Option<(i64, i64)>) -> i32 {
    checked(|| {
        assert!(op <= 1 && mask_kind <= 2 && !descriptors.is_null());
        assert_eq!(count, if op == 0 { 6 } else { 10 });
        assert!(scale.is_finite() && (0.0..=1.0).contains(&dropout_p));
        assert!(!causal || mask_kind == 0);
        assert!(!need_mask || (op == 1 && mask_kind == 2));
        let ds = unsafe { crate::abi::slice(descriptors, count) };
        let xs: Vec<View> = ds.iter().map(|d| unsafe { crate::abi::view(d) }).collect();
        let (q, k, v, mask) = (&xs[0], &xs[1], &xs[2], &xs[3]);
        for x in [q, k, v] { packed(x); assert_eq!(x.shape.len(), 4); assert_eq!(x.dtype, q.dtype); }
        assert!(q.dtype <= 2);
        let (b,h,l,d) = (q.shape[0],q.shape[1],q.shape[2],q.shape[3]);
        let (kh,s,dv) = (k.shape[1],k.shape[2],v.shape[3]);
        assert!(h > 0 && kh > 0 && h % kh == 0 && d > 0 && dv > 0);
        assert_eq!(k.shape, vec![b,kh,s,d]);
        assert_eq!(v.shape, vec![b,kh,s,dv]);
        let rows = b.checked_mul(h).and_then(|n| n.checked_mul(l)).expect("attention rows overflow");
        let scores = rows.checked_mul(s).filter(|&n| n <= u32::MAX as usize).expect("attention RNG/index range exceeds U32");
        let (global_heads, head_start) = head_partition.unwrap_or((h as i64, 0));
        assert!(global_heads >= h as i64 && head_start >= 0 && head_start <= global_heads - h as i64);
        let global_scores = (b as i64).checked_mul(global_heads).and_then(|n|n.checked_mul(l as i64))
            .and_then(|n|n.checked_mul(s as i64)).expect("global attention RNG index overflow");
        counter.checked_add((global_scores as u64).div_ceil(4)).expect("attention RNG counter exhausted");
        assert_eq!(mask.dtype, 0);
        if mask_kind != 0 { assert_eq!(mask.len, scores); }
        let stats = &xs[5]; packed(stats);
        assert_eq!(stats.dtype,0); assert_eq!(stats.len, rows*2);
        let output = &xs[4]; packed(output);
        assert_eq!(output.dtype,q.dtype); assert_eq!(output.shape,vec![b,h,l,dv]);
        let c = client();
        let qt = primitives::tensor(q); let kt = primitives::tensor(k); let vt = primitives::tensor(v);
        let mt = primitives::tensor(mask); let ot = primitives::tensor(output); let st = primitives::tensor(stats);
        let grid = RudaCount::Static(u32::try_from(rows).expect("attention grid overflow"),1,1);
        let dim = RudaDim::new_1d(32);
        if op == 0 {
            if rows != 0 {
                unsafe { kernel::forward::launch::<CudaRuntime>(&c,grid,dim,
                    qt.into_array_arg(),kt.into_array_arg(),vt.into_array_arg(),mt.into_tensor_arg(),
                    ot.into_array_arg(),st.into_array_arg(),h as u32,kh as u32,l as u32,s as u32,scale,dropout_p,
                    seed as u32,(seed>>32) as u32,counter as u32,(counter>>32) as u32,
                    global_heads,head_start,
                    d,dv,32,causal,mask_kind,dropout_p != 0.0,primitives::tensor(q).dtype.into()); }
                LAUNCHES.fetch_add(1,Ordering::Relaxed);
            }
        } else {
            for (x,like) in [(&xs[6],q),(&xs[7],k),(&xs[8],v)] {
                packed(x); assert_eq!(x.shape,like.shape); assert_eq!(x.dtype,q.dtype);
            }
            let dm = &xs[9]; packed(dm); assert_eq!(dm.dtype,0);
            if need_mask && mask.len != 0 {
                let span = mask.shape.iter().zip(&mask.strides).map(|(&d,&s)| (d-1)*s).sum::<usize>()+1;
                assert_eq!(dm.len,span,"mask gradient must match compact mask storage");
            }
            let dqt = primitives::tensor(&xs[6]);
            let dkt = primitives::tensor(&xs[7]); let dvt = primitives::tensor(&xs[8]);
            let dmt = primitives::tensor(dm);
            if dropout_p == 0.0 && !need_mask && l == 512 && s == 512 && d == 128 && dv == 128 && rows != 0 {
                let contraction = View::packed(c.empty(rows.checked_mul(4).expect("attention contraction overflow")), vec![rows], 0);
                let ct = primitives::tensor(&contraction);
                let dtype = match q.dtype {0=>DType::F32,1=>DType::F16,2=>DType::BF16,_=>unreachable!()};
                unsafe {
                    if dm.len != 0 {
                        kernel::zero::launch::<CudaRuntime>(&c,
                            RudaCount::Static(dm.len.div_ceil(128) as u32,1,1),RudaDim::new_1d(128),dmt.into_array_arg());
                    }
                    kernel::backward_query_cached::launch::<CudaRuntime>(&c,grid,RudaDim::new_1d(128),
                        qt.clone().into_array_arg(),kt.clone().into_array_arg(),vt.clone().into_array_arg(),mt.clone().into_tensor_arg(),
                        ot.clone().into_array_arg(),st.clone().into_array_arg(),dqt.into_array_arg(),ct.clone().into_array_arg(),
                        h as u32,kh as u32,scale,causal,mask_kind,dtype.into());
                    kernel::backward_kv_owned::launch::<CudaRuntime>(&c,
                        RudaCount::Static((k.len / 128) as u32,1,1),RudaDim::new_1d(128),
                        qt.into_array_arg(),kt.into_array_arg(),vt.into_array_arg(),mt.into_tensor_arg(),
                        ot.into_array_arg(),st.into_array_arg(),ct.into_array_arg(),dkt.into_array_arg(),dvt.into_array_arg(),
                        h as u32,kh as u32,scale,causal,mask_kind,dtype.into());
                }
                LAUNCHES.fetch_add(2 + u64::from(dm.len != 0),Ordering::Relaxed);
                finish_dispatch(&c);
                return;
            }
            let wk = View::packed(c.empty(k.len.max(1)*4),k.shape.clone(),0);
            let wv = View::packed(c.empty(v.len.max(1)*4),v.shape.clone(),0);
            let wkt = primitives::tensor(&wk); let wvt = primitives::tensor(&wv);
            for x in [&wkt,&wvt,&dmt] {
                let n = x.meta.num_elements();
                if n != 0 { unsafe { kernel::zero::launch::<CudaRuntime>(&c,
                    RudaCount::Static(n.div_ceil(128) as u32,1,1),RudaDim::new_1d(128),x.clone().into_array_arg()); }
                    LAUNCHES.fetch_add(1,Ordering::Relaxed); }
            }
            if rows != 0 {
                unsafe { kernel::backward::launch::<CudaRuntime>(&c,grid,dim,
                    qt.into_array_arg(),kt.into_array_arg(),vt.into_array_arg(),mt.into_tensor_arg(),
                    ot.into_array_arg(),st.into_array_arg(),dqt.into_array_arg(),
                    wkt.clone().into_array_arg(),wvt.clone().into_array_arg(),dmt.into_array_arg(),
                    h as u32,kh as u32,l as u32,s as u32,scale,dropout_p,
                    seed as u32,(seed>>32) as u32,counter as u32,(counter>>32) as u32,
                    global_heads,head_start,
                    d,dv,32,causal,mask_kind,dropout_p != 0.0,need_mask,
                    match q.dtype {0=>DType::F32,1=>DType::F16,2=>DType::BF16,_=>unreachable!()}.into()); }
                LAUNCHES.fetch_add(1,Ordering::Relaxed);
            }
            for (input,out) in [(wkt,dkt),(wvt,dvt)] {
                let n = out.meta.num_elements();
                if n != 0 { unsafe { kernel::narrow::launch::<CudaRuntime>(&c,
                    RudaCount::Static(n.div_ceil(128) as u32,1,1),RudaDim::new_1d(128),
                    input.into_array_arg(),out.into_array_arg(),match q.dtype {0=>DType::F32,1=>DType::F16,2=>DType::BF16,_=>unreachable!()}.into()); }
                    LAUNCHES.fetch_add(1,Ordering::Relaxed); }
            }
        }
        finish_dispatch(&c);
    })
}
