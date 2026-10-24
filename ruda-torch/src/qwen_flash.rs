use super::*;
use rudnn::attention::kernel_ir::{definition::{AttentionGlobalTypes,AttentionOptions,AccumulatorPrecision},
    launch::{self,Strategy,BlueprintStrategy},routines::blackbox_accelerated::BlackboxAcceleratedStrategy};
use ruda_core::tensor::DType;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_qwen2_flash(descriptors:*const Descriptor,count:usize,causal:bool,scale:f32)->i32 {
    checked(|| {
        assert_eq!(count,4);
        let xs:Vec<_>=unsafe {abi::slice(descriptors,count)}.iter().map(|x|unsafe {abi::view(x)}).collect();
        let (q,k,v,o)=(&xs[0],&xs[1],&xs[2],&xs[3]);
        for x in &xs {operation_graph::packed(x);assert_eq!(x.shape.len(),4);assert_eq!(x.dtype,q.dtype);}
        assert!(matches!(q.dtype,1|2) && q.shape[3]>0 && q.shape[2]>0 && k.shape[2]>0);
        assert_eq!(q.shape[0],k.shape[0]);assert!(k.shape[1]>0 && q.shape[1]%k.shape[1]==0);
        assert_eq!(k.shape,v.shape);assert_eq!(q.shape[3],k.shape[3]);
        assert_eq!(o.shape,q.shape);assert!(!causal || q.shape[2]==k.shape[2]);
        assert_eq!(scale,((q.shape[3] as f64).sqrt().recip()) as f32);
        let c=client();let width=q.shape[3];
        let stride=width+8;let temporary_stride=usize::max(stride,72);
        let bytes=|rows:usize|(rows*stride+128*stride+rows*72)*2+rows*temporary_stride*4+rows*4;
        let query_block=if q.shape[2]>=128 && c.properties().hardware.max_shared_memory_size>=bytes(128) {128} else {64};
        let shared_bytes=bytes(query_block);
        if width%16==0 && width<=128 && q.shape[2]>=64
            && xs.iter().all(|x|x.len<=u32::MAX as usize)
            && c.properties().hardware.max_shared_memory_size>=shared_bytes
        {
            macro_rules! run {($ty:ty)=>{unsafe {super::qwen_prefill::flash::launch::<$ty,CudaRuntime>(&c,
                RudaCount::Static(q.shape[2].div_ceil(query_block) as u32,q.shape[1] as u32,q.shape[0] as u32),RudaDim::new_2d(32,(query_block/16) as u32),
                primitives::tensor(q).into_array_arg(),primitives::tensor(k).into_array_arg(),primitives::tensor(v).into_array_arg(),
                primitives::tensor(o).into_array_arg(),q.shape[2] as u32,k.shape[2] as u32,q.shape[1] as u32,k.shape[1] as u32,scale,width,query_block,causal)}};}
            match q.dtype {1=>run!(f16),2=>run!(bf16),_=>unreachable!()}
            LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);return;
        }
        let mut expanded=Vec::new();
        if q.shape[1]!=k.shape[1] {
            for source in [k,v] {
                let shape=vec![q.shape[0],q.shape[1],source.shape[2],width];
                let length=shape.iter().try_fold(1usize,|n,&x|n.checked_mul(x)).unwrap();
                assert!(length<=u32::MAX as usize);
                let destination=View::packed(c.empty(length.checked_mul(element_bytes(q.dtype)).unwrap()),shape,q.dtype);
                macro_rules! copy {($ty:ty)=>{unsafe {super::qwen_prefill::repeat_heads::launch::<$ty,CudaRuntime>(&c,
                    RudaCount::Static(length.div_ceil(128) as u32,1,1),RudaDim::new_1d(128),primitives::tensor(source).into_array_arg(),
                    primitives::tensor(&destination).into_array_arg(),q.shape[1] as u32,source.shape[1] as u32,(source.shape[2]*width) as u32)}};}
                match q.dtype {1=>copy!(f16),2=>copy!(bf16),_=>unreachable!()}
                expanded.push(destination);LAUNCHES.fetch_add(1,Ordering::Relaxed);
            }
        }
        let (k,v)=if expanded.is_empty() {(k,v)} else {(&expanded[0],&expanded[1])};
        let types=AttentionGlobalTypes {
            query:primitives::tensor(q).dtype.into(),key:primitives::tensor(k).dtype.into(),
            value:primitives::tensor(v).dtype.into(),mask:DType::U8.into(),out:primitives::tensor(o).dtype.into(),
        };
        launch::launch_ref::<CudaRuntime>(Strategy::BlackboxAccelerated(BlueprintStrategy::Inferred(
            BlackboxAcceleratedStrategy{num_planes:4,seq_q:1,seq_kv:4})),&c,
            primitives::tensor(q).binding(),primitives::tensor(k).binding(),primitives::tensor(v).binding(),None,
            primitives::tensor(o).binding(),&types,AttentionOptions{causal,
                accumulator_precision:AccumulatorPrecision::Strict(DType::F32.into())})
            .expect("BF16/F16 Tensor Core prefill attention failed");
        LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);
    })
}
