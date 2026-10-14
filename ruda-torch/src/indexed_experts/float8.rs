use super::*;
use ruda_core::{ir::{ElemType,FloatKind,features::MmaConfig},tensor::DType};
use ruda_kernel::{quantization::{float8::encode_scaled,packed::decode_code},tensor::allocation::empty_device_contiguous_dtype};

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_indexed_float8_expert_api_version()->u32 {1}

pub(crate) fn validate(vs:&[View],ints:&[i64],statistics:bool) {
    assert_eq!(ints.len(),if statistics {3}else{7});let count=ints[1];
    assert!((1..=16).contains(&count) && ints[0]>=0 && ints[0]<=u32::MAX as i64-count);
    let count=count as usize;
    assert_eq!(vs.len(),if statistics {count+3}else if ints[2]==2 {2*count+4}else{count+5});
    for v in vs {crate::operation_graph::packed(v);}
    let (x,ids,w)=(&vs[0],&vs[1],&vs[2]);
    assert!(x.shape.len()==2 && x.shape[1]>0 && x.dtype<=2 && matches!(ids.dtype,4|5));
    assert_eq!(ids.shape,vec![x.shape[0]]);assert!(w.shape.len()==2 && w.shape.iter().all(|&n|n>0));
    for weight in &vs[2..2+count] {assert_eq!(weight.shape,w.shape);assert!(weight.dtype<=2);}
    if statistics {
        assert!(matches!(ints[2],0|1));assert_eq!(vs[count+2].shape,vec![count,2,3]);assert_eq!(vs[count+2].dtype,0);
    }else {
        let mode=ints[2];assert!(matches!(mode,0|1|2) && matches!(ints[3],18|19) && matches!(ints[4],18|19));
        assert!((0..=2).contains(&ints[5]) && ints[6]>=0 && ints[6]<(1i64<<count));
        let g=&vs[count+2];let scales=&vs[count+3];assert_eq!(scales.shape,vec![count,2]);assert_eq!(scales.dtype,0);
        assert_eq!(x.shape[1],w.shape[if mode==1 {0}else{1}]);
        if mode==2 {assert_eq!(g.shape,vec![x.shape[0],w.shape[0]]);assert!(g.dtype<=2);}
        else {assert_eq!(g.shape,vec![0]);assert_eq!(g.dtype,x.dtype);}
        for (e,out) in vs[count+4..].iter().enumerate() {
            assert_eq!(out.dtype,0);
            let shape=if mode==2 {if ints[6]&(1<<e)!=0 {w.shape.clone()}else{vec![0]}}
                else {vec![x.shape[0],w.shape[if mode==0 {0}else{1}]]};
            assert_eq!(out.shape,shape);
        }
    }
}

pub(crate) fn prepare_statistics(c:&ComputeClient<CudaRuntime>,vs:&[View],ints:&[i64])->GroupedKernelPlan<CudaRuntime> {
    validate(vs,ints,true);let ts:Vec<_>=vs.iter().map(|v|{let mut t=primitives::tensor(v);t.client=c.clone();t}).collect();
    let count=ints[1] as usize;let mut rows=tensorcore::prepare_rows(c,&ts[1],ints[0] as u32,count);
    let out=&ts[count+2];let width=vs[0].shape[1];
    for e in 0..count {
        for role in 0..2 {
            let value=&ts[if role==0 {0}else{2+e}];
            let blocks=value.meta.num_elements().div_ceil(4096).clamp(1,1024);
            let partials=empty_device_contiguous_dtype(c.clone(),value.device.clone(),[blocks,2].into(),DType::F32);
            unsafe {
                rows.plan.kernels.push(partial_statistics::prepare::<CudaRuntime>(c,RudaCount::Static(blocks as u32,1,1),RudaDim::new_1d(32),
                    value.clone().into_array_arg(),rows.indices.clone().into_array_arg(),rows.offsets.clone().into_array_arg(),
                    partials.clone().into_array_arg(),e as u32,width as u32,(blocks*32) as u32,role==0,role==0 || ints[2]!=0,value.dtype.into()));
                rows.plan.kernels.push(finish_statistics::prepare::<CudaRuntime>(c,RudaCount::Static(1,1,1),RudaDim::new_1d(32),
                    partials.clone().into_array_arg(),rows.offsets.clone().into_array_arg(),out.clone().into_array_arg(),
                    e as u32,role as u32,role==0 || ints[2]!=0));
            }
            rows.plan.workspace.push(partials);
        }
    }
    rows.plan
}

pub(crate) fn prepare_product(c:&ComputeClient<CudaRuntime>,vs:&[View],ints:&[i64])->GroupedKernelPlan<CudaRuntime> {
    validate(vs,ints,false);let ts:Vec<_>=vs.iter().map(|v|{let mut t=primitives::tensor(v);t.client=c.clone();t}).collect();
    let count=ints[1] as usize;let mode=ints[2] as u32;let (inner,columns)=(vs[2].shape[1],vs[2].shape[0]);
    let format_a=ints[3] as u32;let format_b=ints[4] as u32;
    let dtype=|format|ElemType::Float(if format==18 {FloatKind::E4M3}else{FloatKind::E5M2}).into();
    let native=ints[5]!=2 && c.features().matmul.mma.contains(&MmaConfig {
        a_type:dtype(format_a),b_type:dtype(format_b),cd_type:DType::F32.into(),m:16,n:8,k:32});
    assert!(ints[5]!=1 || native,"native indexed FP8 MMA is unavailable");
    let h=&c.properties().hardware;
    assert!(h.plane_size_min==32 && h.plane_size_max==32);
    assert!(native || h.max_shared_memory_size>=2048 && c.features().matmul.cmma.contains(&MmaConfig {
        a_type:DType::F16.into(),b_type:DType::F16.into(),cd_type:DType::F32.into(),m:16,n:16,k:16}),
        "indexed FP8 GEMM requires native FP8 MMA or exact FP16 carriers");
    let tile_n=if native {8}else{16};let n=if mode==0 {columns}else{inner};
    assert!(n.div_ceil(tile_n)<=h.max_ruda_count.0 as usize);
    let mut rows=tensorcore::prepare_rows(c,&ts[1],ints[0] as u32,count);
    if mode!=2 && vs[count+4].len!=0 {
        rows.plan.kernels.push(unsafe {tensorcore::zero::prepare::<CudaRuntime>(c,
            RudaCount::Static(vs[count+4].len.div_ceil(128) as u32,1,1),RudaDim::new_1d(128),ts[count+4].clone().into_array_arg(),DType::F32.into())});
    }
    for e in 0..count {
        let out=&ts[count+4+if mode==2 {e}else{0}];if out.meta.num_elements()==0 {continue;}
        let a=&ts[if mode==2 {count+2}else{0}];let b=&ts[if mode==2 {0}else{2+e}];
        let m=if mode==2 {columns}else{vs[0].shape[0]};
        let blocks=m.div_ceil(16).min(128).min(h.max_ruda_count.1 as usize);assert!(blocks>0);
        let grid=RudaCount::Static(n.div_ceil(tile_n) as u32,blocks as u32,1);
        unsafe {
            let node=if native {native_product::prepare::<CudaRuntime>(c,grid,RudaDim::new_1d(32),
                a.clone().into_array_arg(),b.clone().into_array_arg(),rows.indices.clone().into_array_arg(),rows.offsets.clone().into_array_arg(),
                ts[count+3].clone().into_array_arg(),out.clone().into_array_arg(),e as u32,inner as u32,columns as u32,
                (blocks*16) as u32,mode,format_a,format_b,a.dtype.into(),b.dtype.into(),dtype(format_a),dtype(format_b))}
            else {carrier_product::prepare::<CudaRuntime>(c,grid,RudaDim::new_1d(32),
                a.clone().into_array_arg(),b.clone().into_array_arg(),rows.indices.clone().into_array_arg(),rows.offsets.clone().into_array_arg(),
                ts[count+3].clone().into_array_arg(),out.clone().into_array_arg(),e as u32,inner as u32,columns as u32,
                (blocks*16) as u32,mode,format_a,format_b,a.dtype.into(),b.dtype.into())};
            rows.plan.kernels.push(node);
        }
    }
    rows.plan
}

#[ruda(launch)]
fn partial_statistics<F:Float>(value:&Array<F>,indices:&Array<u32>,offsets:&Array<u32>,out:&mut Array<f32>,
    expert:u32,width:u32,step:u32,#[comptime] routed:bool,#[comptime] enabled:bool,#[define(F)] _dtype:StorageType) {
    let begin=offsets[expert] as usize;let count=(offsets[expert+1]-offsets[expert]) as usize;
    let mut length=0usize;
    if comptime!(enabled) {if count!=0 {length=if comptime!(routed) {count*width as usize}else{value.len()};}}
    let mut p=RUDA_POS_X as usize*32+UNIT_POS as usize;let mut maximum=0.0f32;let mut invalid=0.0f32;
    while p<length {
        let at=if comptime!(routed) {indices[begin+p/width as usize] as usize*width as usize+p%width as usize}else{p};
        let v=f32::cast_from(value[at]).abs();
        if v.to_bits()>=0x7f800000u32 {invalid=1.0;}else {maximum=f32::max(maximum,v);}
        if length-p<=step as usize {p=length;}else {p+=step as usize;}
    }
    maximum=plane_max(maximum);invalid=plane_max(invalid);
    if UNIT_POS==0 {out[RUDA_POS_X*2]=maximum;out[RUDA_POS_X*2+1]=invalid;}
}

#[ruda(launch)]
fn finish_statistics(partials:&Array<f32>,offsets:&Array<u32>,out:&mut Array<f32>,expert:u32,role:u32,#[comptime] enabled:bool) {
    let mut p=UNIT_POS as usize;let mut maximum=0.0f32;let mut invalid=0.0f32;
    while p<partials.len()/2 {maximum=f32::max(maximum,partials[p*2]);invalid=f32::max(invalid,partials[p*2+1]);p+=32;}
    maximum=plane_max(maximum);invalid=plane_max(invalid);
    if UNIT_POS==0 {
        let at=(expert*2+role)*3;out[at]=if invalid==0.0 {maximum}else{0.0};out[at+1]=invalid;
        out[at+2]=if comptime!(enabled) {f32::cast_from(offsets[expert+1]>offsets[expert])}else{0.0};
    }
}

#[ruda]
pub(super) fn left<F:Float>(value:&Array<F>,indices:&Array<u32>,begin:usize,row:usize,k:usize,inner:usize,columns:usize,#[comptime] mode:u32)->f32 {
    let at=if comptime!(mode==2) {indices[begin+k] as usize*columns+row}
        else {indices[begin+row] as usize*(if comptime!(mode==0) {inner}else{columns})+k};
    f32::cast_from(value[at])
}

#[ruda]
pub(super) fn right<F:Float>(value:&Array<F>,indices:&Array<u32>,begin:usize,column:usize,k:usize,inner:usize,#[comptime] mode:u32)->f32 {
    let at=if comptime!(mode==2) {indices[begin+k] as usize*inner+column}
        else if comptime!(mode==1) {k*inner+column}else {column*inner+k};
    f32::cast_from(value[at])
}

#[ruda(launch)]
fn carrier_product<L:Float,R:Float>(a:&Array<L>,b:&Array<R>,indices:&Array<u32>,offsets:&Array<u32>,scales:&Array<f32>,out:&mut Array<f32>,
    expert:u32,inner:u32,columns:u32,step:u32,#[comptime] mode:u32,#[comptime] format_a:u32,#[comptime] format_b:u32,
    #[define(L)] _left:StorageType,#[define(R)] _right:StorageType) {
    let begin=offsets[expert] as usize;let count=(offsets[expert+1]-offsets[expert]) as usize;
    let m=if comptime!(mode==2) {columns as usize}else{count};
    let n=if comptime!(mode==0) {columns as usize}else{inner as usize};
    let k=if comptime!(mode==2) {count}else if comptime!(mode==1) {columns as usize}else{inner as usize};
    let sa=scales[expert*2];let sb=scales[expert*2+1];let lane=UNIT_POS as usize;let col=RUDA_POS_X as usize*16;
    let mut row=RUDA_POS_Y as usize*16;
    let mut al=SharedMemory::<f16>::new_aligned(256usize,32usize);let mut bl=SharedMemory::<f16>::new_aligned(256usize,32usize);
    let mut result=SharedMemory::<f32>::new_aligned(256usize,32usize);
    while row<m {
        let acc=cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,16usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0f32);
        let mut base=0usize;
        while base<k {
            #[unroll] for i in 0usize..8usize {
                let t=lane+i*32;let r=t/16;let c=t%16;let mut av=0.0f32;let mut bv=0.0f32;
                if row+r<m && base+c<k {av=decode_code(encode_scaled(left(a,indices,begin,row+r,base+c,inner as usize,columns as usize,mode),sa,format_a) as u32,format_a);}
                if base+r<k && col+c<n {bv=decode_code(encode_scaled(right(b,indices,begin,col+c,base+r,inner as usize,mode),sb,format_b) as u32,format_b);}
                al[t]=f16::cast_from(av);bl[t]=f16::cast_from(bv);
            }
            sync_ruda();
            let aa=cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::A,16usize,16usize,16usize,cmma::MatrixLayout::RowMajor,&al.to_slice(),16);
            let bb=cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::B,16usize,16usize,16usize,cmma::MatrixLayout::RowMajor,&bl.to_slice(),16);
            cmma::execute::<f16,f16,f32,f32>(&aa,&bb,&acc,&acc);sync_ruda();
            if k-base<=16 {base=k;}else {base+=16;}
        }
        cmma::store(&mut result.to_slice_mut(),&acc,16,cmma::MatrixLayout::RowMajor);sync_ruda();
        #[unroll] for i in 0usize..8usize {
            let t=lane+i*32;let rr=row+t/16;let cc=col+t%16;
            if rr<m && cc<n {let destination=if comptime!(mode==2) {rr}else{indices[begin+rr] as usize};out[destination*n+cc]=result[t]*sa*sb;}
        }
        sync_ruda();if m-row<=step as usize {row=m;}else {row+=step as usize;}
    }
}

#[ruda(launch)]
fn native_product<L:Float,R:Float,A:Float,B:Float>(a:&Array<L>,b:&Array<R>,indices:&Array<u32>,offsets:&Array<u32>,scales:&Array<f32>,out:&mut Array<f32>,
    expert:u32,inner:u32,columns:u32,step:u32,#[comptime] mode:u32,#[comptime] format_a:u32,#[comptime] format_b:u32,
    #[define(L)] _left:StorageType,#[define(R)] _right:StorageType,#[define(A)] _a_type:StorageType,#[define(B)] _b_type:StorageType) {
    let begin=offsets[expert] as usize;let count=(offsets[expert+1]-offsets[expert]) as usize;
    let m=if comptime!(mode==2) {columns as usize}else{count};let n=if comptime!(mode==0) {columns as usize}else{inner as usize};
    let k=if comptime!(mode==2) {count}else if comptime!(mode==1) {columns as usize}else{inner as usize};
    let sa=scales[expert*2];let sb=scales[expert*2+1];let lane=UNIT_POS_PLANE;let col=RUDA_POS_X as usize*8;
    let def=cmma::MmaDefinition::<A,B,f32>::new(16usize,8usize,32usize);
    let va=def.vector_size(cmma::MatrixIdent::A);let vb=def.vector_size(cmma::MatrixIdent::B);let vc=def.vector_size(cmma::MatrixIdent::Accumulator);
    let size!(NA)=va;let size!(NB)=vb;let size!(NC)=vc;
    let ca=def.vectors_per_lane(cmma::MatrixIdent::A);let cb=def.vectors_per_lane(cmma::MatrixIdent::B);let cc=def.vectors_per_lane(cmma::MatrixIdent::Accumulator);
    let mut ra=Array::<Vector<A,NA>>::new(ca);let mut rb=Array::<Vector<B,NB>>::new(cb);let mut rc=Array::<Vector<f32,NC>>::new(cc);
    let mut row=RUDA_POS_Y as usize*16;
    while row<m {
        #[unroll] for i in 0..cc {rc[i]=Vector::cast_from(0.0f32);}
        let mut base=0usize;
        while base<k {
            #[unroll] for i in 0..ca {
                let mut reg=Vector::empty();
                #[unroll] for j in 0..va {
                    let (r,c)=def.position_of_nth(lane,(i*va+j) as u32,cmma::MatrixIdent::A);let mut code=0u8;
                    if row+(r as usize)<m && base+(c as usize)<k {
                        code=encode_scaled(left(a,indices,begin,row+r as usize,base+c as usize,inner as usize,columns as usize,mode),sa,format_a);
                    }
                    reg[j]=A::reinterpret(code);
                }
                ra[i]=reg;
            }
            #[unroll] for i in 0..cb {
                let mut reg=Vector::empty();
                #[unroll] for j in 0..vb {
                    let (r,c)=def.position_of_nth(lane,(i*vb+j) as u32,cmma::MatrixIdent::B);let mut code=0u8;
                    if base+(r as usize)<k && col+(c as usize)<n {
                        code=encode_scaled(right(b,indices,begin,col+c as usize,base+r as usize,inner as usize,mode),sb,format_b);
                    }
                    reg[j]=B::reinterpret(code);
                }
                rb[i]=reg;
            }
            def.execute_inplace(&ra,&rb,&mut rc);if k-base<=32 {base=k;}else {base+=32;}
        }
        #[unroll] for i in 0..cc {
            let reg=rc[i];
            #[unroll] for j in 0..vc {
                let (r,c)=def.position_of_nth(lane,(i*vc+j) as u32,cmma::MatrixIdent::Accumulator);let rr=row+r as usize;let column=col+c as usize;
                if rr<m && column<n {let destination=if comptime!(mode==2) {rr}else{indices[begin+rr] as usize};out[destination*n+column]=reg[j]*sa*sb;}
            }
        }
        if m-row<=step as usize {row=m;}else {row+=step as usize;}
    }
}
