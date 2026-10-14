use crate::dsl as kernel_dsl;
use crate::dsl::prelude::*;
use half::f16;
use crate::tensor::{RudaTensor, allocation::empty_device_contiguous_dtype};
use ruda_core::{device::Device, tensor::{DType, Shape}, ir::{ElemType, FloatKind, features::MmaConfig}};
use super::packed::{decode_code, encode_code};

/// Byte-encoded FP8 GEMM. Scales are scalar, or per output row/column.
/// Native FP8 MMA is selected where available; FP16 tensor cores otherwise
/// carry the FP8 values exactly, with FP32 accumulation and post-GEMM scaling.
pub fn scaled_mm<R:Runtime>(a:&RudaTensor<R>,b:&RudaTensor<R>,sa:&RudaTensor<R>,sb:&RudaTensor<R>,
    format_a:u32,format_b:u32,transpose_a:bool,transpose_b:bool,strategy:u32)->RudaTensor<R> {
    assert!(matches!(format_a,18|19) && matches!(format_b,18|19) && strategy<=2);
    assert_eq!(a.dtype,DType::U8);assert_eq!(b.dtype,DType::U8);
    assert_eq!(sa.dtype,DType::F32);assert_eq!(sb.dtype,DType::F32);
    for x in [a,b,sa,sb] {
        assert!(x.is_contiguous() && x.qparams.is_none());
        assert_eq!(x.device.to_id(),a.device.to_id());
        assert!(x.client.same_execution_queue(&a.client));
        assert!(x.meta.num_elements()<=u32::MAX as usize);
    }
    assert_eq!(a.meta.num_dims(),2);assert_eq!(b.meta.num_dims(),2);
    let ash=a.meta.shape();let bsh=b.meta.shape();
    let (m,k)=if transpose_a {(ash[1],ash[0])} else {(ash[0],ash[1])};
    let (kb,n)=if transpose_b {(bsh[1],bsh[0])} else {(bsh[0],bsh[1])};
    assert_eq!(k,kb);
    assert!(sa.meta.num_elements()==1 || sa.meta.shape().as_slice()==[m,1]);
    assert!(sb.meta.num_elements()==1 || sb.meta.shape().as_slice()==[1,n]);
    assert!(m.checked_mul(n).expect("FP8 output overflow")<=u32::MAX as usize);
    let ty=|fmt|ElemType::Float(if fmt==18 {FloatKind::E4M3} else {FloatKind::E5M2}).into();
    let native_cfg=MmaConfig{a_type:ty(format_a),b_type:ty(format_b),
        cd_type:ElemType::Float(FloatKind::F32).into(),m:16,n:8,k:32};
    let native=a.client.features().matmul.mma.contains(&native_cfg) && strategy!=2;
    assert!(strategy!=1 || native,"native FP8 MMA is unavailable on this device");
    let cfg=MmaConfig{a_type:DType::F16.into(),b_type:DType::F16.into(),
        cd_type:ElemType::Float(FloatKind::F32).into(),m:16,n:16,k:16};
    let h=&a.client.properties().hardware;
    let tile_n=if native {8} else {16};
    assert!(h.plane_size_min==32 && h.plane_size_max==32
        && n.div_ceil(tile_n)<=h.max_ruda_count.0 as usize && m.div_ceil(16)<=h.max_ruda_count.1 as usize);
    assert!(native || (h.max_shared_memory_size>=2048 && a.client.features().matmul.cmma.contains(&cfg)),
        "FP8 GEMM needs native FP8 or FP16 tensor cores");
    let out=empty_device_contiguous_dtype(a.client.clone(),a.device.clone(),Shape::new([m,n]),DType::F32);
    if m==0 || n==0 {return out;}
    let grid=RudaCount::Static(n.div_ceil(tile_n) as u32,m.div_ceil(16) as u32,1);
    if native {
        native_mm::launch::<R>(&a.client,grid,RudaDim::new_1d(32),
            a.clone().into_array_arg(),b.clone().into_array_arg(),sa.clone().into_array_arg(),sb.clone().into_array_arg(),
            out.clone().into_array_arg(),m as u32,n as u32,k as u32,transpose_a,transpose_b,
            sa.meta.num_elements()==1,sb.meta.num_elements()==1,ty(format_a),ty(format_b));
    } else {
        carrier_mm::launch::<R>(&a.client,grid,RudaDim::new_1d(32),
            a.clone().into_array_arg(),b.clone().into_array_arg(),sa.clone().into_array_arg(),sb.clone().into_array_arg(),
            out.clone().into_array_arg(),m as u32,n as u32,k as u32,transpose_a,transpose_b,
            sa.meta.num_elements()==1,sb.meta.num_elements()==1,format_a,format_b);
    }
    out
}

#[ruda]
fn address(row:u32,col:u32,rows:u32,cols:u32,#[comptime] transpose:bool)->usize {
    if comptime!(transpose) {(col*rows+row) as usize} else {(row*cols+col) as usize}
}

pub fn prepare_scaled_mm<R:Runtime>(a:&RudaTensor<R>,b:&RudaTensor<R>,sa:&RudaTensor<R>,sb:&RudaTensor<R>,out:&RudaTensor<R>,
    format_a:u32,format_b:u32,ta:bool,tb:bool,strategy:u32)->PreparedKernel<R> {
    assert!(matches!(format_a,18|19) && matches!(format_b,18|19) && strategy<=2);
    assert_eq!(a.dtype,DType::U8);assert_eq!(b.dtype,DType::U8);
    for t in [sa,sb,out] {assert_eq!(t.dtype,DType::F32);}
    for t in [a,b,sa,sb,out] {
        assert!(t.is_contiguous() && t.qparams.is_none() && t.meta.num_elements()<=u32::MAX as usize);
        assert_eq!(t.device.to_id(),a.device.to_id());assert!(t.client.same_execution_queue(&a.client));
    }
    assert_eq!(a.meta.num_dims(),2);assert_eq!(b.meta.num_dims(),2);
    let ash=a.meta.shape();let bsh=b.meta.shape();
    let (m,k)=if ta {(ash[1],ash[0])} else {(ash[0],ash[1])};
    let (kb,n)=if tb {(bsh[1],bsh[0])} else {(bsh[0],bsh[1])};
    assert_eq!(k,kb);assert!(m>0 && n>0);assert_eq!(out.meta.shape().as_slice(),&[m,n]);
    assert!(sa.meta.num_elements()==1 || sa.meta.shape().as_slice()==[m,1]);
    assert!(sb.meta.num_elements()==1 || sb.meta.shape().as_slice()==[1,n]);
    let ty=|fmt|ElemType::Float(if fmt==18 {FloatKind::E4M3} else {FloatKind::E5M2}).into();
    let native_cfg=MmaConfig{a_type:ty(format_a),b_type:ty(format_b),cd_type:ElemType::Float(FloatKind::F32).into(),m:16,n:8,k:32};
    let native=strategy!=2 && a.client.features().matmul.mma.contains(&native_cfg);
    assert!(strategy!=1 || native,"native FP8 graph MMA unavailable");
    let carrier=MmaConfig{a_type:DType::F16.into(),b_type:DType::F16.into(),cd_type:ElemType::Float(FloatKind::F32).into(),m:16,n:16,k:16};
    let h=&a.client.properties().hardware;let tile_n=if native {8} else {16};
    assert!(h.plane_size_min==32 && h.plane_size_max==32 && n.div_ceil(tile_n)<=h.max_ruda_count.0 as usize
        && m.div_ceil(16)<=h.max_ruda_count.1 as usize && (native || (h.max_shared_memory_size>=2048 && a.client.features().matmul.cmma.contains(&carrier))));
    let grid=RudaCount::Static(n.div_ceil(tile_n) as u32,m.div_ceil(16) as u32,1);
    unsafe {if native {native_mm::prepare::<R>(&a.client,grid,RudaDim::new_1d(32),a.clone().into_array_arg(),b.clone().into_array_arg(),
        sa.clone().into_array_arg(),sb.clone().into_array_arg(),out.clone().into_array_arg(),m as u32,n as u32,k as u32,ta,tb,
        sa.meta.num_elements()==1,sb.meta.num_elements()==1,ty(format_a),ty(format_b))}
    else {carrier_mm::prepare::<R>(&a.client,grid,RudaDim::new_1d(32),a.clone().into_array_arg(),b.clone().into_array_arg(),
        sa.clone().into_array_arg(),sb.clone().into_array_arg(),out.clone().into_array_arg(),m as u32,n as u32,k as u32,ta,tb,
        sa.meta.num_elements()==1,sb.meta.num_elements()==1,format_a,format_b)}}
}

pub fn prepare_current_quantize<R:Runtime>(x:&RudaTensor<R>,encoded:&RudaTensor<R>,scale:&RudaTensor<R>,partials:&RudaTensor<R>,
    format:u32,margin:u32,power_of_two:bool)->Vec<PreparedKernel<R>> {
    assert!(matches!(format,18|19) && margin<=16 && matches!(x.dtype,DType::F32|DType::F16|DType::BF16));
    assert_eq!(encoded.dtype,DType::U8);assert_eq!(scale.dtype,DType::F32);assert_eq!(partials.dtype,DType::F32);
    assert_eq!(encoded.meta.shape(),x.meta.shape());assert_eq!(scale.meta.num_elements(),1);
    let blocks=x.meta.num_elements().div_ceil(4096).max(1);assert_eq!(partials.meta.shape().as_slice(),&[blocks,2]);
    for t in [x,encoded,scale,partials] {
        assert!(t.is_contiguous() && t.qparams.is_none() && t.meta.num_elements()<=u32::MAX as usize);
        assert_eq!(t.device.to_id(),x.device.to_id());assert!(t.client.same_execution_queue(&x.client));
    }
    let maximum=if format==18 {448.0f32} else {57344.0f32};
    let maximum=maximum*2.0f32.powi(-(margin as i32));
    let mut nodes=vec![unsafe {amax_partials::prepare::<R>(&x.client,RudaCount::Static(blocks as u32,1,1),RudaDim::new_1d(32),
        x.clone().into_array_arg(),partials.clone().into_array_arg(),x.dtype.into())},
        unsafe {current_scale::prepare::<R>(&x.client,RudaCount::Static(1,1,1),RudaDim::new_1d(32),
        partials.clone().into_array_arg(),scale.clone().into_array_arg(),maximum,power_of_two)}];
    if x.meta.num_elements()>0 {nodes.push(unsafe {quantize::prepare::<R>(&x.client,
        RudaCount::Static(x.meta.num_elements().div_ceil(128) as u32,1,1),RudaDim::new_1d(128),
        x.clone().into_array_arg(),scale.clone().into_array_arg(),encoded.clone().into_array_arg(),format,x.dtype.into())});}
    nodes
}

#[ruda(launch)]
fn amax_partials<F:Float>(x:&Array<F>,out:&mut Array<f32>,#[define(F)] _dtype:StorageType) {
    let block=RUDA_POS_X as usize;let lane=UNIT_POS as usize;let base=block*4096;
    let mut maximum=0.0f32;let mut nan=0.0f32;
    for iteration in 0usize..128usize {
        let p=base+lane+iteration*32;
        if p<x.len() {
            let value=f32::cast_from(x[p]).abs();
            if value!=value {nan=1.0;} else {maximum=f32::max(maximum,value);}
        }
    }
    maximum=plane_max(maximum);nan=plane_max(nan);
    if lane==0 {out[block*2]=maximum;out[block*2+1]=nan;}
}

#[ruda(launch)]
fn current_scale(partials:&Array<f32>,out:&mut Array<f32>,maximum:f32,#[comptime] power_of_two:bool) {
    let lane=UNIT_POS as usize;let mut i=lane;let mut amax=0.0f32;let mut nan=0.0f32;
    while i<partials.len()/2 {amax=f32::max(amax,partials[i*2]);nan=f32::max(nan,partials[i*2+1]);i+=32;}
    amax=plane_max(amax);nan=plane_max(nan);
    if lane==0 {
        let mut scale=1.0f32;
        if nan==0.0 && amax>0.0 && amax.to_bits()<0x7f800000u32 {scale=f32::max(amax/maximum,f32::from_bits(0x00800000u32));}
        if comptime!(power_of_two) {
            let bits=scale.to_bits();let mut exponent=bits&0x7f800000u32;
            if bits&0x007fffffu32!=0u32 {exponent+=0x00800000u32;}
            scale=f32::from_bits(exponent);
        }
        out[0]=scale;
    }
}

#[ruda(launch)]
fn native_mm<A:Float,B:Float>(a:&Array<u8>,b:&Array<u8>,sa:&Array<f32>,sb:&Array<f32>,out:&mut Array<f32>,
    m:u32,n:u32,k:u32,#[comptime] ta:bool,#[comptime] tb:bool,#[comptime] scalar_a:bool,#[comptime] scalar_b:bool,
    #[define(A)] _a_type:StorageType,#[define(B)] _b_type:StorageType) {
    let def=cmma::MmaDefinition::<A,B,f32>::new(16usize,8usize,32usize);
    let va=def.vector_size(cmma::MatrixIdent::A);let vb=def.vector_size(cmma::MatrixIdent::B);
    let vc=def.vector_size(cmma::MatrixIdent::Accumulator);
    let size!(NA)=va;let size!(NB)=vb;let size!(NC)=vc;
    let ca=def.vectors_per_lane(cmma::MatrixIdent::A);
    let cb=def.vectors_per_lane(cmma::MatrixIdent::B);
    let cc=def.vectors_per_lane(cmma::MatrixIdent::Accumulator);
    let mut ra=Array::<Vector<A,NA>>::new(ca);let mut rb=Array::<Vector<B,NB>>::new(cb);
    let mut rc=Array::<Vector<f32,NC>>::new(cc);
    #[unroll] for i in 0..cc {rc[i]=Vector::cast_from(0.0f32);}
    let row_base=RUDA_POS_Y*16;let col_base=RUDA_POS_X*8;let lane=UNIT_POS_PLANE;
    let mut base=0u32;
    while base<k {
        #[unroll] for i in 0..ca {
            let mut reg=Vector::empty();
            #[unroll] for j in 0..va {
                let (r,c)=def.position_of_nth(lane,(i*va+j) as u32,cmma::MatrixIdent::A);
                let row=row_base+r;let col=base+c;let mut code=0u8;
                if row<m && col<k {code=a[address(row,col,m,k,ta)];}
                reg[j]=A::reinterpret(code);
            }
            ra[i]=reg;
        }
        #[unroll] for i in 0..cb {
            let mut reg=Vector::empty();
            #[unroll] for j in 0..vb {
                let (r,c)=def.position_of_nth(lane,(i*vb+j) as u32,cmma::MatrixIdent::B);
                let row=base+r;let col=col_base+c;let mut code=0u8;
                if row<k && col<n {code=b[address(row,col,k,n,tb)];}
                reg[j]=B::reinterpret(code);
            }
            rb[i]=reg;
        }
        def.execute_inplace(&ra,&rb,&mut rc);
        if k-base<=32 {base=k;} else {base+=32;}
    }
    #[unroll] for i in 0..cc {
        let reg=rc[i];
        #[unroll] for j in 0..vc {
            let (r,c)=def.position_of_nth(lane,(i*vc+j) as u32,cmma::MatrixIdent::Accumulator);
            let row=row_base+r;let col=col_base+c;
            if row<m && col<n {
                let mut ai=row as usize;let mut bi=col as usize;
                if comptime!(scalar_a) {ai=0;} if comptime!(scalar_b) {bi=0;}
                out[(row*n+col) as usize]=reg[j]*sa[ai]*sb[bi];
            }
        }
    }
}

#[ruda(launch)]
fn carrier_mm(a:&Array<u8>,b:&Array<u8>,sa:&Array<f32>,sb:&Array<f32>,out:&mut Array<f32>,
    m:u32,n:u32,k:u32,#[comptime] ta:bool,#[comptime] tb:bool,#[comptime] scalar_a:bool,#[comptime] scalar_b:bool,
    #[comptime] format_a:u32,#[comptime] format_b:u32) {
    let row_base=RUDA_POS_Y*16;let col_base=RUDA_POS_X*16;let lane=UNIT_POS as usize;
    let mut left=SharedMemory::<f16>::new_aligned(256usize,32usize);
    let mut right=SharedMemory::<f16>::new_aligned(256usize,32usize);
    let mut result=SharedMemory::<f32>::new_aligned(256usize,32usize);
    let acc=cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,16usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0f32);
    let mut base=0u32;
    while base<k {
        #[unroll] for i in 0usize..8usize {
            let t=lane+i*32;let r=t as u32/16;let c=t as u32%16;
            let mut av=0.0f32;let mut bv=0.0f32;
            if row_base+r<m && base+c<k {av=decode_code(a[address(row_base+r,base+c,m,k,ta)] as u32,format_a);}
            if base+r<k && col_base+c<n {bv=decode_code(b[address(base+r,col_base+c,k,n,tb)] as u32,format_b);}
            left[t]=f16::cast_from(av);right[t]=f16::cast_from(bv);
        }
        sync_ruda();
        let aa=cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::A,16usize,16usize,16usize,cmma::MatrixLayout::RowMajor,&left.to_slice(),16u32);
        let bb=cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::B,16usize,16usize,16usize,cmma::MatrixLayout::RowMajor,&right.to_slice(),16u32);
        cmma::execute::<f16,f16,f32,f32>(&aa,&bb,&acc,&acc);sync_ruda();
        if k-base<=16 {base=k;} else {base+=16;}
    }
    cmma::store(&mut result.to_slice_mut(),&acc,16,cmma::MatrixLayout::RowMajor);sync_ruda();
    #[unroll] for i in 0usize..8usize {
        let t=lane+i*32;let row=row_base+t as u32/16;let col=col_base+t as u32%16;
        if row<m && col<n {
            let mut ai=row as usize;let mut bi=col as usize;
            if comptime!(scalar_a) {ai=0;} if comptime!(scalar_b) {bi=0;}
            out[(row*n+col) as usize]=result[t]*sa[ai]*sb[bi];
        }
    }
}

#[ruda(launch)]
pub fn quantize<F:Float>(x:&Array<F>,scale:&Array<f32>,out:&mut Array<u8>,#[comptime] format:u32,
    #[define(F)] _dtype:StorageType) {
    let i=ABSOLUTE_POS as usize;
    if i<out.len() {
        let value=f32::cast_from(x[i])/scale[0];
        let mut code=encode_code(value,format,true);
        if (value.to_bits()&0x7fffffffu32)>=0x7f800000u32 {code=encode_code(value,format,false);}
        out[i]=code as u8;
    }
}
