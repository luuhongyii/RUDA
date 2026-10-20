use super::*;
use rudnn::ring_attention as kernel;

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_ring_attention_api_version() -> u32 { 1 }

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_indexed_ring_attention_api_version() -> u32 { 1 }

fn contiguous(view: &View) {
    let mut stride = 1usize;
    for axis in (0..view.shape.len()).rev() {
        assert!(view.shape[axis] <= 1 || view.strides[axis] == stride);
        stride = stride.checked_mul(view.shape[axis]).expect("ring attention shape overflow");
    }
    assert!(view.len <= u32::MAX as usize, "ring block exceeds U32 storage indexing");
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_ring_attention(op: u32, descriptors: *const Descriptor,
    count: usize, batch: usize, global_keys: u32, causal: bool, left: i64, right: i64,
    scale: f32, p: f32, softcap: f32, seed: u64, counter: u64) -> i32 {
    checked(|| {
        assert!(op <= 2 && batch > 0 && left >= -1 && right >= -1);
        assert!(scale.is_finite() && (0.0..=1.0).contains(&p) && softcap.is_finite() && softcap >= 0.0);
        assert_eq!(count, match op { 0 => 7, 1 => 11, _ => 3 });
        let descriptors = unsafe { crate::abi::slice(descriptors, count) };
        let xs: Vec<_> = descriptors.iter().map(|d| unsafe { crate::abi::view(d) }).collect();
        for x in &xs { contiguous(x); }
        let c = client();
        let ts: Vec<_> = xs.iter().map(primitives::tensor).collect();
        if op == 2 {
            assert_eq!(xs[0].dtype, 0); assert_eq!(xs[1].dtype, 0); assert!(xs[2].dtype <= 2);
            assert_eq!(xs[0].shape.len(), 3); assert_eq!(xs[2].shape, xs[0].shape);
            let width = xs[0].shape[2];
            assert!(width > 0);
            assert_eq!(xs[1].shape, vec![xs[0].shape[0] * xs[0].shape[1], 2]);
            if xs[0].len != 0 {
                unsafe { kernel::normalize::launch::<CudaRuntime>(&c,
                    RudaCount::Static(xs[0].len.div_ceil(128) as u32, 1, 1), RudaDim::new_1d(128),
                    ts[0].clone().into_array_arg(), ts[1].clone().into_array_arg(), ts[2].clone().into_array_arg(),
                    width as u32, ts[2].dtype.into()); }
                LAUNCHES.fetch_add(1, Ordering::Relaxed);
            }
            finish_dispatch(&c);
            return;
        }
        let (q, k, v) = (&xs[0], &xs[1], &xs[2]);
        assert!(q.dtype <= 2);
        for x in [q, k, v] { assert_eq!(x.shape.len(), 3); assert_eq!(x.dtype, q.dtype); }
        let (queries, heads, d) = (q.shape[0], q.shape[1], q.shape[2]);
        let (keys, kv_heads, dv) = (k.shape[0], k.shape[1], v.shape[2]);
        assert!(heads > 0 && kv_heads > 0 && heads % kv_heads == 0 && d > 0 && dv > 0);
        assert!(queries % batch == 0 && keys % batch == 0 && keys / batch <= global_keys as usize);
        assert_eq!(k.shape, vec![keys, kv_heads, d]); assert_eq!(v.shape, vec![keys, kv_heads, dv]);
        assert_eq!(xs[3].dtype, 4); assert_eq!(xs[4].dtype, 4);
        assert_eq!(xs[3].shape.len(), 2); assert_eq!(xs[3].shape[0], queries);
        assert!(xs[3].shape[1] == 3 || xs[3].shape[1] == 4);
        assert_eq!(xs[4].shape, vec![keys, 3]);
        let indexed = xs[3].shape[1] == 4;
        let rows = queries.checked_mul(heads).expect("ring launch size overflow");
        let output = if op == 0 { 5 } else { 6 };
        let statistics = if op == 0 { 6 } else { 7 };
        assert_eq!(xs[output].shape, vec![queries, heads, dv]); assert_eq!(xs[output].dtype, 0);
        assert_eq!(xs[statistics].shape, vec![rows, 2]); assert_eq!(xs[statistics].dtype, 0);
        if op == 1 {
            assert_eq!(xs[5].shape, xs[output].shape); assert_eq!(xs[5].dtype, q.dtype);
            for (index, like) in [(8, q), (9, k), (10, v)] {
                assert_eq!(xs[index].shape, like.shape); assert_eq!(xs[index].dtype, 0);
            }
        }
        if rows != 0 {
            let grid = RudaCount::Static(u32::try_from(rows).expect("ring launch exceeds U32"), 1, 1);
            let dim = RudaDim::new_1d(32);
            if op == 0 {
                unsafe { kernel::forward::launch::<CudaRuntime>(&c, grid, dim,
                    ts[0].clone().into_array_arg(), ts[1].clone().into_array_arg(), ts[2].clone().into_array_arg(),
                    ts[3].clone().into_array_arg(), ts[4].clone().into_array_arg(),
                    ts[5].clone().into_array_arg(), ts[6].clone().into_array_arg(),
                    heads as u32, kv_heads as u32, (queries / batch) as u32, (keys / batch) as u32, global_keys,
                    scale, p, softcap, left, right, seed as u32, (seed >> 32) as u32, counter as u32, (counter >> 32) as u32,
                    d, dv, 32, causal, p != 0.0, softcap != 0.0, indexed, ts[0].dtype.into()); }
            } else {
                unsafe { kernel::backward::launch::<CudaRuntime>(&c, grid, dim,
                    ts[0].clone().into_array_arg(), ts[1].clone().into_array_arg(), ts[2].clone().into_array_arg(),
                    ts[3].clone().into_array_arg(), ts[4].clone().into_array_arg(),
                    ts[5].clone().into_array_arg(), ts[6].clone().into_array_arg(), ts[7].clone().into_array_arg(),
                    ts[8].clone().into_array_arg(), ts[9].clone().into_array_arg(), ts[10].clone().into_array_arg(),
                    heads as u32, kv_heads as u32, (queries / batch) as u32, (keys / batch) as u32, global_keys,
                    scale, p, softcap, left, right, seed as u32, (seed >> 32) as u32, counter as u32, (counter >> 32) as u32,
                    d, dv, 32, causal, p != 0.0, softcap != 0.0, indexed, ts[0].dtype.into()); }
            }
            LAUNCHES.fetch_add(1, Ordering::Relaxed);
        }
        finish_dispatch(&c);
    })
}
