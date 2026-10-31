use super::*;
use ruda_kernel::dsl as kernel_dsl;

fn matrix_view(source:&View,row:usize,column:usize)->View {
    let axes=(0..source.shape.len()).filter(|&d|d!=row && d!=column).chain([row,column]).collect::<Vec<_>>();
    View{handle:source.handle.clone(),shape:axes.iter().map(|&d|source.shape[d]).collect(),
        strides:axes.iter().map(|&d|source.strides[d]).collect(),len:source.len,dtype:source.dtype}
}

fn common_batch_views(a:&View,b:&View,out:&View)->Option<(View,View,View)> {
    let mut boundaries=vec![1usize,out.shape[0]];
    for source in [a,b] {
        let axis=source.shape.len()-2;let batches:usize=source.shape[..axis].iter().product();
        let mut suffix=out.shape[0]/batches;boundaries.push(suffix);
        for &size in source.shape[..axis].iter().rev() {suffix*=size;boundaries.push(suffix);}
    }
    boundaries.sort_unstable();boundaries.dedup();
    if boundaries.windows(2).any(|v|v[1]%v[0]!=0) {return None;}
    let mut batch_shape=boundaries.windows(2).rev().map(|v|v[1]/v[0]).collect::<Vec<_>>();
    let expand=|source:&View|->Option<View> {
        let axis=source.shape.len()-2;let batches:usize=source.shape[..axis].iter().product();
        let mut factors=source.shape[..axis].iter().copied().zip(source.strides[..axis].iter().copied())
            .chain([(out.shape[0]/batches,0)]).filter(|&(size,_)|size>1).collect::<Vec<_>>();
        let mut position=0usize;let mut strides=Vec::with_capacity(batch_shape.len()+2);
        for &size in &batch_shape {
            let (extent,stride)=*factors.get(position)?;
            if extent%size!=0 {return None;}
            let tail=extent/size;strides.push(stride*tail);
            if tail==1 {position+=1;}else {factors[position]=(tail,stride);}
        }
        if position!=factors.len() {return None;}
        let mut shape=batch_shape.clone();shape.extend_from_slice(&source.shape[axis..]);
        strides.extend_from_slice(&source.strides[axis..]);let len=shape.iter().product();
        Some(View{handle:source.handle.clone(),shape,strides,len,dtype:source.dtype})
    };
    let mut left=expand(a)?;let mut right=expand(b)?;
    for axis in (1..batch_shape.len()).rev() {
        if left.strides[axis-1]==batch_shape[axis]*left.strides[axis]
            && right.strides[axis-1]==batch_shape[axis]*right.strides[axis] {
            let factor=batch_shape[axis];batch_shape[axis-1]*=factor;batch_shape.remove(axis);
            for view in [&mut left,&mut right] {
                let factor=view.shape[axis];view.shape[axis-1]*=factor;view.shape.remove(axis);view.strides.remove(axis-1);
            }
        }
    }
    if batch_shape.len()>6 {return None;}
    batch_shape.extend_from_slice(&out.shape[1..]);
    Some((left,right,View::packed(out.handle.clone(),batch_shape,out.dtype)))
}

pub(super) fn prepare_grouped(c:&ComputeClient<CudaRuntime>,a:&View,b:&View,out:&View,code:u32)
    ->PreparedKernel<CudaRuntime> {
    let a=matrix_view(a,(code&7) as usize,((code>>3)&7) as usize);
    let b=matrix_view(b,((code>>6)&7) as usize,((code>>9)&7) as usize);
    prepare_batched_views(c,&a,&b,out)
}

pub(super) fn prepare_layout(c:&ComputeClient<CudaRuntime>,a:&View,b:&View,out:&View,code:usize)
    ->PreparedKernel<CudaRuntime> {
    let variants=super::graph_contract::bmm_layout_variants(a.shape.len());
    let transform=|source:&View,layout:usize| {
        let (axes,repeat)=super::graph_contract::bmm_layout_axes(source.shape.len(),layout).expect("validated BMM layout");
        let batch:usize=axes[..axes.len()-2].iter().map(|&d|source.shape[d]).product();
        let mut shape=axes.iter().map(|&d|source.shape[d]).collect::<Vec<_>>();
        let mut strides=axes.iter().map(|&d|source.strides[d]).collect::<Vec<_>>();
        if out.shape[0]!=batch {shape.insert(repeat,out.shape[0]/batch);strides.insert(repeat,0);}
        let len=shape.iter().product();
        View{handle:source.handle.clone(),shape,strides,len,dtype:source.dtype}
    };
    let left=transform(a,code%variants);let right=transform(b,code/variants);
    prepare_batched_views(c,&left,&right,out)
}

fn prepare_batched_views(c:&ComputeClient<CudaRuntime>,a:&View,b:&View,out:&View)
    ->PreparedKernel<CudaRuntime> {
    if let Some((left,right,output))=common_batch_views(a,b,out) {
        let bind=|v:&View| {let mut t=primitives::tensor(v);t.client=c.clone();t};
        if a.dtype==0 {
            if let Some(kernel)=rublas::tensor_matmul::prepare_matmul_f32_tiled(&bind(&left),&bind(&right),&bind(&output)) {return kernel;}
        } else {
            match rublas::tensor_matmul::prepare_matmul_pipelined(bind(&left),bind(&right),bind(&output),rublas::tensor_matmul::F32MathMode::Strict) {
                Ok(kernel)=>return kernel,
                Err(rublas::kernel_ir::definition::MatmulSetupError::Unavailable(_)
                    |rublas::kernel_ir::definition::MatmulSetupError::InvalidConfig(_)
                    |rublas::kernel_ir::definition::MatmulSetupError::Vectorization(_))=>{},
                Err(error)=>panic!("RUDA grouped graph matmul preparation failed: {error}"),
            }
        }
    }
    let (m,k,n)=(a.shape[a.shape.len()-2],a.shape[a.shape.len()-1],b.shape[b.shape.len()-1]);
    if a.dtype!=0 {
        if let Some(kernel)=prepare_strided_tile(c,a,b,out,m,k,n) {return kernel;}
    }
    if let Some(kernel)=prepare_residue(c,a,b,out,m,k,n) {return kernel;}
    let blocks=u32::try_from(out.len.div_ceil(128)).expect("grouped BMM grid overflow");
    let aa=a.shape.len()-2;let ba=b.shape.len()-2;
    let ag=out.shape[0]/a.shape[..aa].iter().product::<usize>();
    let bg=out.shape[0]/b.shape[..ba].iter().product::<usize>();
    macro_rules! run {($f:ty)=>{unsafe {grouped_scalar::prepare::<$f,CudaRuntime>(c,RudaCount::Static(blocks,1,1),RudaDim::new_1d(128),
        a.arg(),b.arg(),out.arg(),m,k,n,aa,ba,ag,bg)}};}
    match a.dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),_=>unreachable!()}
}

#[ruda]
fn batch_offset<F:Float>(source:&Tensor<F>,batch:usize,#[comptime] axis:usize,#[comptime] group:usize)->usize {
    let mut index=batch/group;let mut offset=0usize;
    #[unroll] for step in 0usize..axis {
        let d=comptime!(axis-1-step);offset+=index%source.shape(d)*source.stride(d);index/=source.shape(d);
    }
    offset
}

#[ruda(launch)]
fn grouped_scalar<F:Float+RudaElement>(a:&Tensor<F>,b:&Tensor<F>,out:&mut Tensor<F>,m:usize,k:usize,n:usize,
    #[comptime] aa:usize,#[comptime] ba:usize,#[comptime] ag:usize,#[comptime] bg:usize) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let batch=p/(m*n);let row=p/n%m;let col=p%n;
        let ab=batch_offset(a,batch,aa,ag);let bb=batch_offset(b,batch,ba,bg);let mut sum=0.0f32;
        for j in 0usize..k {sum=fma(f32::cast_from(a[ab+row*a.stride(aa)+j*a.stride(aa+1)]),
            f32::cast_from(b[bb+j*b.stride(ba)+col*b.stride(ba+1)]),sum);}
        out[p]=F::cast_from(sum);
    }
}

pub(super) fn prepare(c:&ComputeClient<CudaRuntime>,a:&View,b:&View,out:&View)
    ->Option<PreparedKernel<CudaRuntime>> {
    use ruda_core::ir::{ElemType,FloatKind,features::MmaConfig};
    if a.dtype>2 {return None;}
    let rank=a.shape.len();
    let (m,k,n)=(a.shape[rank-2],a.shape[rank-1],b.shape[rank-1]);
    if k==0 {return None;}
    let bind=|v:&View| {
        let mut tensor=primitives::tensor(v);
        tensor.client=c.clone();
        tensor
    };
    if a.dtype==0 {
        if let Some(kernel)=rublas::tensor_matmul::prepare_matmul_f32_tiled(&bind(a),&bind(b),&bind(out)) {
            return Some(kernel);
        }
        return prepare_residue(c,a,b,out,m,k,n);
    }
    if rank==2 && (1..=4).contains(&m) {
        match rublas::tensor_matmul::prepare_gemv_low_precision(bind(a),bind(b),bind(out)) {
            Ok(kernel)=>return Some(kernel),
            Err(rublas::kernel_ir::definition::MatmulSetupError::Unavailable(_)
                |rublas::kernel_ir::definition::MatmulSetupError::InvalidConfig(_)
                |rublas::kernel_ir::definition::MatmulSetupError::Vectorization(_))=>{},
            Err(error)=>panic!("RUDA graph GEMV preparation failed: {error}"),
        }
    }
    let prepared=rublas::tensor_matmul::prepare_matmul_pipelined(bind(a),bind(b),bind(out),
        rublas::tensor_matmul::F32MathMode::Strict);
    match prepared {
        Ok(kernel)=>return Some(kernel),
        Err(rublas::kernel_ir::definition::MatmulSetupError::Unavailable(_)
            |rublas::kernel_ir::definition::MatmulSetupError::InvalidConfig(_)
            |rublas::kernel_ir::definition::MatmulSetupError::Vectorization(_))=>{},
        Err(error)=>panic!("RUDA graph matmul preparation failed: {error}"),
    }
    let batches=if rank==3 {a.shape[0]} else {1};
    if !a.contiguous() || !b.contiguous() {
        return prepare_strided_tile(c,a,b,out,m,k,n).or_else(||prepare_residue(c,a,b,out,m,k,n));
    }
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

fn prepare_strided_tile(c:&ComputeClient<CudaRuntime>,a:&View,b:&View,out:&View,m:usize,k:usize,n:usize)
    ->Option<PreparedKernel<CudaRuntime>> {
    use ruda_core::ir::{ElemType,FloatKind,features::MmaConfig};
    let batched=out.shape.len()==3;let batches=if batched {out.shape[0]}else {1};
    let aa=a.shape.len()-2;let ba=b.shape.len()-2;
    let ag=batches/a.shape[..aa].iter().product::<usize>();let bg=batches/b.shape[..ba].iter().product::<usize>();
    let storage=ElemType::Float(if a.dtype==1 {FloatKind::F16}else {FloatKind::BF16}).into();
    let cfg=MmaConfig{a_type:storage,b_type:storage,cd_type:ElemType::Float(FloatKind::F32).into(),m:16,n:16,k:16};
    let h=&c.properties().hardware;
    if h.plane_size_min!=32 || h.plane_size_max!=32 || h.max_units_per_ruda<128 || h.max_ruda_dim.0<128 || h.max_shared_memory_size<8192
        || !c.features().matmul.cmma.contains(&cfg) || n.div_ceil(32)>h.max_ruda_count.0 as usize
        || m.div_ceil(32)>h.max_ruda_count.1 as usize || batches>h.max_ruda_count.2 as usize {return None;}
    let grid=RudaCount::Static(n.div_ceil(32) as u32,m.div_ceil(32) as u32,batches as u32);
    macro_rules! run {($f:ty)=>{unsafe {strided_tile::prepare::<$f,CudaRuntime>(c,grid,RudaDim::new_1d(128),
        a.arg(),b.arg(),out.arg(),m,n,k,batched,aa,ba,ag,bg)}};}
    Some(if a.dtype==1 {run!(f16)}else {run!(bf16)})
}

#[ruda(launch)]
fn strided_tile<F:Float+RudaElement>(a:&Tensor<F>,b:&Tensor<F>,out:&mut Tensor<F>,m:usize,n:usize,k:usize,
    #[comptime] batched:bool,#[comptime] aa:usize,#[comptime] ba:usize,#[comptime] ag:usize,#[comptime] bg:usize)
{
    let tid=UNIT_POS as usize;let warp=tid/32;
    let row_base=RUDA_POS_Y as usize*32;let col_base=RUDA_POS_X as usize*32;
    let axis=comptime!(if batched {1usize}else {0usize});
    let ab=batch_offset(a,RUDA_POS_Z as usize,aa,ag);let bb=batch_offset(b,RUDA_POS_Z as usize,ba,bg);
    let mut ob=0usize;if comptime!(batched) {ob=RUDA_POS_Z as usize*out.stride(0);}
    let mut left=SharedMemory::<F>::new_aligned(1024usize,32usize);
    let mut right=SharedMemory::<F>::new_aligned(1024usize,32usize);
    let mut result=SharedMemory::<f32>::new_aligned(1024usize,32usize);
    let acc=cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,16usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0f32);
    let mut base=0usize;
    while base<k {
        #[unroll] for step in 0usize..8 {
            let p=tid+step*128;let row=p/32;let col=p%32;
            let mut x=F::cast_from(0);let mut y=F::cast_from(0);
            if row_base+row<m && base+col<k {x=a[ab+(row_base+row)*a.stride(aa)+(base+col)*a.stride(aa+1)];}
            if base+row<k && col_base+col<n {y=b[bb+(base+row)*b.stride(ba)+(col_base+col)*b.stride(ba+1)];}
            left[p]=x;right[p]=y;
        }
        sync_ruda();
        #[unroll] for half in 0usize..2 {
            let left_start=(warp/2)*16*32+half*16;let right_start=half*16*32+(warp%2)*16;
            let aa=cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::A,16usize,16usize,16usize,cmma::MatrixLayout::RowMajor,
                &left.slice(left_start,1024usize),32u32);
            let bb=cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::B,16usize,16usize,16usize,cmma::MatrixLayout::RowMajor,
                &right.slice(right_start,1024usize),32u32);
            cmma::execute::<F,F,f32,f32,_>(&aa,&bb,&acc,&acc);
        }
        sync_ruda();if k-base<=32 {base=k;}else {base+=32;}
    }
    let start=(warp/2)*16*32+(warp%2)*16;
    cmma::store(&mut result.slice_mut(start,1024usize),&acc,32u32,cmma::MatrixLayout::RowMajor);sync_ruda();
    #[unroll] for step in 0usize..8 {
        let p=tid+step*128;let row=row_base+p/32;let col=col_base+p%32;
        if row<m && col<n {out[ob+row*out.stride(axis)+col*out.stride(axis+1)]=F::cast_from(result[p]);}
    }
}

fn prepare_residue(c:&ComputeClient<CudaRuntime>,a:&View,b:&View,out:&View,m:usize,k:usize,n:usize)
    ->Option<PreparedKernel<CudaRuntime>> {
    let batched=out.shape.len()==3;let batches=if batched {out.shape[0]}else{1};
    let aa=a.shape.len()-2;let ba=b.shape.len()-2;
    let ag=batches/a.shape[..aa].iter().product::<usize>();let bg=batches/b.shape[..ba].iter().product::<usize>();
    let h=&c.properties().hardware;
    if h.max_units_per_ruda<256 || h.max_shared_memory_size<2048
        || n.div_ceil(16)>h.max_ruda_count.0 as usize || m.div_ceil(16)>h.max_ruda_count.1 as usize
        || batches>h.max_ruda_count.2 as usize {return None;}
    let grid=RudaCount::Static(n.div_ceil(16) as u32,m.div_ceil(16) as u32,batches as u32);
    macro_rules! run {($ty:ty)=>{unsafe {residue::prepare::<$ty,CudaRuntime>(c,grid,RudaDim::new_1d(256),
        a.arg(),b.arg(),out.arg(),m as u32,k as u32,n as u32,batched,aa,ba,ag,bg)}};}
    Some(match a.dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),_=>return None})
}

#[ruda(launch)]
fn residue<F:Float+RudaElement>(a:&Tensor<F>,b:&Tensor<F>,out:&mut Tensor<F>,m:u32,k:u32,n:u32,
    #[comptime] batched:bool,#[comptime] aa:usize,#[comptime] ba:usize,#[comptime] ag:usize,#[comptime] bg:usize) {
    let tid=UNIT_POS as usize;let local_row=tid/16usize;let local_col=tid%16usize;
    let row=RUDA_POS_Y as usize*16usize+local_row;let col=RUDA_POS_X as usize*16usize+local_col;
    let axis=comptime!(if batched {1usize}else{0usize});
    let ab=batch_offset(a,RUDA_POS_Z as usize,aa,ag);let bb=batch_offset(b,RUDA_POS_Z as usize,ba,bg);
    let mut ob=0usize;if comptime!(batched) {ob=RUDA_POS_Z as usize*out.stride(0);}
    let mut left=SharedMemory::<f32>::new(256usize);let mut right=SharedMemory::<f32>::new(256usize);
    let mut sum=0.0f32;let mut base=0usize;
    while base<k as usize {
        let mut x=0.0f32;let mut y=0.0f32;
        if row<m as usize && base+local_col<k as usize {x=f32::cast_from(a[ab+row*a.stride(aa)+(base+local_col)*a.stride(aa+1)]);}
        if base+local_row<k as usize && col<n as usize {y=f32::cast_from(b[bb+(base+local_row)*b.stride(ba)+col*b.stride(ba+1)]);}
        left[tid]=x;right[tid]=y;sync_ruda();
        #[unroll] for j in 0usize..16usize {sum=fma(left[local_row*16+j],right[j*16+local_col],sum);}
        sync_ruda();base+=16usize;
    }
    if row<m as usize && col<n as usize {out[ob+row*out.stride(axis)+col*out.stride(axis+1)]=F::cast_from(sum);}
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
