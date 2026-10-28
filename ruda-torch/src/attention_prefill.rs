use super::*;

pub(super) fn prepare(c:&ComputeClient<CudaRuntime>,q:&View,k:&View,v:&View,out:&View,
    mask:&View,stats:&View,mask_kind:u32,causal:bool,scale:f32)->Option<PreparedKernel<CudaRuntime>> {
    if q.shape.len()!=4 || q.shape[0]==0 || q.shape[1]==0 || !matches!(q.dtype,1|2) || k.dtype!=q.dtype || v.dtype!=q.dtype
        || q.shape[2]<64 || q.shape[3]==0 || q.shape[3]>128 || q.shape[3]%16!=0
        || k.shape[2]==0 || k.shape!=v.shape || k.shape[3]!=q.shape[3]
        || k.shape[1]==0 || q.shape[1]%k.shape[1]!=0 || mask_kind>2
        || [q,k,v,out,mask,stats].iter().any(|x|x.len>u32::MAX as usize) {return None;}
    use ruda_core::ir::{ElemType,FloatKind,features::MmaConfig};
    let storage=ElemType::Float(if q.dtype==1 {FloatKind::F16}else {FloatKind::BF16}).into();
    let h=&c.properties().hardware;
    if h.plane_size_min==32 && h.plane_size_max==32 && h.max_units_per_ruda>=128 && h.max_ruda_dim.1>=4
        && h.max_shared_memory_size>=2*64*(q.shape[3]+8)*2
        && q.shape[2].div_ceil(64)<=h.max_ruda_count.0 as usize
        && q.shape[1]<=h.max_ruda_count.1 as usize && q.shape[0]<=h.max_ruda_count.2 as usize
        && c.features().matmul.ldmatrix.contains(&storage)
        && c.features().matmul.mma.contains(&MmaConfig{a_type:storage,b_type:storage,
            cd_type:ElemType::Float(FloatKind::F32).into(),m:16,n:8,k:16}) {
        let asynchronous=c.features().copy_async && c.features().alignment
            && rublas::kernel_ir::components::global::read::validate_async_barrier(c.properties()).is_ok()
            && h.max_shared_memory_size>=2*64*(q.shape[3]+8)*2+32
            && [q,k,v].iter().all(|x|x.handle.offset_start.unwrap_or(0)%16==0);
        let tensor=|x:&View| {let mut t=primitives::tensor(x);t.client=c.clone();t};
        let wide=asynchronous && mask_kind==0 && scale>0.0 && q.shape[2]>=256
            && std::env::var("RUDA_PREFILL").map_or(true,|value|value!="legacy")
            && h.max_shared_memory_size>=(128+128)*(q.shape[3]+8)*2+32;
        if wide {
            macro_rules! run_wide {($ty:ty)=>{super::attention_wide::flash_wide::prepare::<$ty,CudaRuntime>(c,
                RudaCount::Static(q.shape[2].div_ceil(128) as u32,q.shape[1] as u32,q.shape[0] as u32),
                RudaDim::new_2d(32,4),tensor(q).into_array_arg(),tensor(k).into_array_arg(),tensor(v).into_array_arg(),
                tensor(out).into_array_arg(),tensor(stats).into_array_arg(),q.shape[2] as u32,k.shape[2] as u32,
                q.shape[1] as u32,k.shape[1] as u32,scale,scale*core::f32::consts::LOG2_E,q.shape[3],causal)};}
            return Some(if q.dtype==1 {run_wide!(f16)}else {run_wide!(bf16)});
        }
        macro_rules! run {($ty:ty)=>{super::attention_mma::flash::prepare::<$ty,CudaRuntime>(c,
            RudaCount::Static(q.shape[2].div_ceil(64) as u32,q.shape[1] as u32,q.shape[0] as u32),
            RudaDim::new_2d(32,4),tensor(q).into_array_arg(),tensor(k).into_array_arg(),tensor(v).into_array_arg(),
            tensor(out).into_array_arg(),mask.arg(),tensor(stats).into_array_arg(),q.shape[2] as u32,k.shape[2] as u32,
            q.shape[1] as u32,k.shape[1] as u32,scale,q.shape[3],causal,mask_kind,asynchronous)};}
        return Some(if q.dtype==1 {run!(f16)}else {run!(bf16)});
    }
    if !c.features().matmul.cmma.contains(&MmaConfig{a_type:storage,b_type:storage,
        cd_type:ElemType::Float(FloatKind::F32).into(),m:16,n:16,k:16}) {return None;}
    let width=q.shape[3];let stride=width+8;let temporary_stride=usize::max(stride,72);
    let bytes=|rows:usize|(rows*stride+128*stride+rows*72)*2+rows*temporary_stride*4+rows*4;
    let hardware=&c.properties().hardware;
    let block=if q.shape[2]>=128 && hardware.max_shared_memory_size>=bytes(128) {128}else {64};
    if hardware.plane_size_min!=32 || hardware.plane_size_max!=32 || hardware.max_shared_memory_size<bytes(block)
        || hardware.max_units_per_ruda<(block*2) as u32 || hardware.max_ruda_dim.1<(block/16) as u32
        || q.shape[2].div_ceil(block)>hardware.max_ruda_count.0 as usize
        || q.shape[1]>hardware.max_ruda_count.1 as usize || q.shape[0]>hardware.max_ruda_count.2 as usize {return None;}
    let tensor=|x:&View| {let mut t=primitives::tensor(x);t.client=c.clone();t};
    macro_rules! run {($ty:ty)=>{super::qwen_prefill::flash::prepare::<$ty,CudaRuntime>(c,
        RudaCount::Static(q.shape[2].div_ceil(block) as u32,q.shape[1] as u32,q.shape[0] as u32),
        RudaDim::new_2d(32,(block/16) as u32),tensor(q).into_array_arg(),tensor(k).into_array_arg(),
        tensor(v).into_array_arg(),tensor(out).into_array_arg(),mask.arg(),tensor(stats).into_array_arg(),
        q.shape[2] as u32,k.shape[2] as u32,q.shape[1] as u32,k.shape[1] as u32,scale,width,block,causal,mask_kind)};}
    Some(match q.dtype {1=>run!(f16),2=>run!(bf16),_=>unreachable!()})
}
