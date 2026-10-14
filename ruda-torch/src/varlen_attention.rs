use super::*;
use rudnn::{dense_attention, varlen_attention as kernel};

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_varlen_attention_api_version() -> u32 { 1 }

fn contiguous(view: &View) {
    let mut stride = 1usize;
    for axis in (0..view.shape.len()).rev() {
        assert!(view.shape[axis] <= 1 || view.strides[axis] == stride);
        stride = stride.checked_mul(view.shape[axis]).expect("varlen shape overflow");
    }
    assert!(view.len <= u32::MAX as usize, "varlen storage exceeds U32 indexing");
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_varlen_attention(op: u32, descriptors: *const Descriptor,
    count: usize, causal: bool, lower_right: bool, left: i64, right: i64,
    scale: f32, p: f32, softcap: f32, seed: u64, counter: u64) -> i32 {
    checked(|| {
        assert!(op <= 1 && left >= -1 && right >= -1);
        assert!(scale.is_finite() && (0.0..=1.0).contains(&p));
        assert!(softcap.is_finite() && softcap >= 0.0);
        assert_eq!(count, if op == 0 { 6 } else { 9 });
        let descriptors = unsafe { crate::abi::slice(descriptors, count) };
        let xs: Vec<_> = descriptors.iter().map(|d| unsafe { crate::abi::view(d) }).collect();
        for x in &xs { contiguous(x); }
        let (q,k,v,schedule) = (&xs[0],&xs[1],&xs[2],&xs[3]);
        assert!(q.dtype <= 2);
        for x in [q,k,v] { assert_eq!(x.shape.len(),3); assert_eq!(x.dtype,q.dtype); }
        let (queries,heads,d) = (q.shape[0],q.shape[1],q.shape[2]);
        let (keys,kv_heads,dv) = (k.shape[0],k.shape[1],v.shape[2]);
        assert!(heads > 0 && kv_heads > 0 && heads % kv_heads == 0 && d > 0 && dv > 0);
        assert_eq!(k.shape,vec![keys,kv_heads,d]);
        assert_eq!(v.shape,vec![keys,kv_heads,dv]);
        assert_eq!(schedule.dtype,4); assert_eq!(schedule.shape.len(),2); assert_eq!(schedule.shape[1],6);
        let documents = u32::try_from(schedule.shape[0]).expect("varlen document count overflow");
        assert!(documents > 0);
        let rows = queries.checked_mul(heads).expect("varlen row count overflow");
        assert_eq!(xs[4].shape,vec![queries,heads,dv]); assert_eq!(xs[4].dtype,q.dtype);
        assert_eq!(xs[5].shape,vec![rows,2]); assert_eq!(xs[5].dtype,0);
        let c = client();
        let ts: Vec<_> = xs.iter().map(primitives::tensor).collect();
        let dtype = ts[0].dtype;
        let grid = RudaCount::Static(u32::try_from(rows).expect("varlen launch overflow"),1,1);
        let dim = RudaDim::new_1d(32);
        if op == 0 {
            if rows != 0 {
                unsafe { kernel::forward::launch::<CudaRuntime>(&c,grid,dim,
                    ts[0].clone().into_array_arg(),ts[1].clone().into_array_arg(),ts[2].clone().into_array_arg(),
                    ts[3].clone().into_array_arg(),ts[4].clone().into_array_arg(),ts[5].clone().into_array_arg(),
                    heads as u32,kv_heads as u32,documents,scale,p,softcap,left,right,
                    seed as u32,(seed>>32) as u32,counter as u32,(counter>>32) as u32,
                    d,dv,32,causal,lower_right,p != 0.0,softcap != 0.0,dtype.into()); }
                LAUNCHES.fetch_add(1,Ordering::Relaxed);
            }
        } else {
            for (x,like) in [(&xs[6],q),(&xs[7],k),(&xs[8],v)] {
                assert_eq!(x.shape,like.shape); assert_eq!(x.dtype,q.dtype);
            }
            let wk = View::packed(c.empty(k.len.max(1)*4),k.shape.clone(),0);
            let wv = View::packed(c.empty(v.len.max(1)*4),v.shape.clone(),0);
            let kt = primitives::tensor(&wk); let vt = primitives::tensor(&wv);
            for x in [&kt,&vt] {
                let n = x.meta.num_elements();
                if n != 0 {
                    unsafe { dense_attention::zero::launch::<CudaRuntime>(&c,
                        RudaCount::Static(n.div_ceil(128) as u32,1,1),RudaDim::new_1d(128),x.clone().into_array_arg()); }
                    LAUNCHES.fetch_add(1,Ordering::Relaxed);
                }
            }
            if rows != 0 {
                unsafe { kernel::backward::launch::<CudaRuntime>(&c,grid,dim,
                    ts[0].clone().into_array_arg(),ts[1].clone().into_array_arg(),ts[2].clone().into_array_arg(),
                    ts[3].clone().into_array_arg(),ts[4].clone().into_array_arg(),ts[5].clone().into_array_arg(),
                    ts[6].clone().into_array_arg(),kt.clone().into_array_arg(),vt.clone().into_array_arg(),
                    heads as u32,kv_heads as u32,documents,scale,p,softcap,left,right,
                    seed as u32,(seed>>32) as u32,counter as u32,(counter>>32) as u32,
                    d,dv,32,causal,lower_right,p != 0.0,softcap != 0.0,dtype.into()); }
                LAUNCHES.fetch_add(1,Ordering::Relaxed);
            }
            for (input,out) in [(kt,ts[7].clone()),(vt,ts[8].clone())] {
                let n = out.meta.num_elements();
                if n != 0 {
                    unsafe { dense_attention::narrow::launch::<CudaRuntime>(&c,
                        RudaCount::Static(n.div_ceil(128) as u32,1,1),RudaDim::new_1d(128),
                        input.into_array_arg(),out.into_array_arg(),dtype.into()); }
                    LAUNCHES.fetch_add(1,Ordering::Relaxed);
                }
            }
        }
        finish_dispatch(&c);
    })
}
