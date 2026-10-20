use super::*;
use ruda_kernel::dsl as kernel_dsl;

pub(super) fn prepare(c:&ComputeClient<CudaRuntime>,a:&View,b:&View,out:&View)
    ->Option<PreparedKernel<CudaRuntime>> {
    use ruda_core::ir::{ElemType,FloatKind,features::MmaConfig};
    if !matches!(a.dtype,1|2) {return None;}
    let rank=a.shape.len();
    let (m,k,n)=(a.shape[rank-2],a.shape[rank-1],b.shape[rank-1]);
    if k==0 {return None;}
    let batches=if rank==3 {a.shape[0]} else {1};
    let storage=ElemType::Float(if a.dtype==1 {FloatKind::F16} else {FloatKind::BF16}).into();
    let cfg=MmaConfig{a_type:storage,b_type:storage,cd_type:ElemType::Float(FloatKind::F32).into(),m:16,n:16,k:16};
    let h=&c.properties().hardware;
    if h.plane_size_min!=32 || h.plane_size_max!=32 || h.max_shared_memory_size<2048
        || !c.features().matmul.cmma.contains(&cfg) || n.div_ceil(16)>h.max_ruda_count.0 as usize
        || m.div_ceil(16)>h.max_ruda_count.1 as usize || batches>h.max_ruda_count.2 as usize {return None;}
    let grid=RudaCount::Static(n.div_ceil(16) as u32,m.div_ceil(16) as u32,batches as u32);
    macro_rules! launch {($ty:ty)=>{unsafe {tile::prepare::<$ty,CudaRuntime>(c,grid,RudaDim::new_1d(32),
        a.arg(),b.arg(),out.arg(),m as u32,n as u32,k as u32)}};}
    Some(if a.dtype==1 {launch!(f16)} else {launch!(bf16)})
}

#[ruda(launch)]
fn tile<F:Float+RudaElement>(a:&Tensor<F>,b:&Tensor<F>,out:&mut Tensor<F>,m:u32,n:u32,k:u32) {
    let batch=RUDA_POS_Z;let row_base=RUDA_POS_Y*16;let col_base=RUDA_POS_X*16;let lane=UNIT_POS;
    let mut left=SharedMemory::<F>::new_aligned(256usize,32usize);
    let mut right=SharedMemory::<F>::new_aligned(256usize,32usize);
    let mut result=SharedMemory::<f32>::new_aligned(256usize,32usize);
    let acc=cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,16usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0f32);
    let mut base=0u32;
    while base<k {
        #[unroll] for step in 0u32..8u32 {
            let t=lane+step*32;let r=t/16;let col=t%16;
            let mut av=F::cast_from(0.0f32);let mut bv=F::cast_from(0.0f32);
            if row_base+r<m && base+col<k {av=a[(batch*m*k+(row_base+r)*k+base+col) as usize];}
            if base+r<k && col_base+col<n {bv=b[(batch*k*n+(base+r)*n+col_base+col) as usize];}
            left[t as usize]=av;right[t as usize]=bv;
        }
        sync_ruda();
        let aa=cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::A,16usize,16usize,16usize,cmma::MatrixLayout::RowMajor,&left.to_slice(),16u32);
        let bb=cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::B,16usize,16usize,16usize,cmma::MatrixLayout::RowMajor,&right.to_slice(),16u32);
        cmma::execute::<F,F,f32,f32, _>(&aa,&bb,&acc,&acc);sync_ruda();
        if k-base<=16 {base=k;} else {base+=16;}
    }
    cmma::store(&mut result.to_slice_mut(),&acc,16,cmma::MatrixLayout::RowMajor);sync_ruda();
    #[unroll] for step in 0u32..8u32 {
        let t=lane+step*32;let row=row_base+t/16;let col=col_base+t%16;
        if row<m && col<n {out[(batch*m*n+row*n+col) as usize]=F::cast_from(result[t as usize]);}
    }
}
