//! Row-packed quantization with explicit affine parameters. Codes are unsigned,
//! least-significant-bit first; each row begins at a byte boundary. Parameters
//! implement `decode(code) * scale + offset`, independently for every row/group.
//! This representation does not change the serialized symmetric QFloat ABI.
use crate::dsl as kernel_dsl;
use crate::dsl::prelude::*;
use crate::tensor::{RudaTensor, allocation::empty_device_contiguous_dtype};
use ruda_core::{device::Device, tensor::{DType, Shape}};
use ruda_core::ir::{ElemType, FloatKind, features::MmaConfig};

/// Integer bit widths, or IEEE-like miniature floating-point encodings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PackedFormat { Int2, Int3, Int4, Int5, Int6, Int8, E2M1, E4M3, E5M2 }
impl PackedFormat {
    pub fn code(self) -> u32 { match self {
        Self::Int2=>2, Self::Int3=>3, Self::Int4=>4, Self::Int5=>5,
        Self::Int6=>6, Self::Int8=>8, Self::E2M1=>14, Self::E4M3=>18, Self::E5M2=>19,
    } }
    pub fn from_code(code:u32) -> Self { match code {
        2=>Self::Int2,3=>Self::Int3,4=>Self::Int4,5=>Self::Int5,6=>Self::Int6,
        8=>Self::Int8,14=>Self::E2M1,18=>Self::E4M3,19=>Self::E5M2,
        _=>panic!("unknown packed quantization format"),
    } }
    pub fn bits(self)->usize { match self { Self::E2M1=>4,Self::E4M3|Self::E5M2=>8,_=>self.code() as usize } }
}

/// Frozen packed tensor, supporting irregular GPTQ input-to-group assignments.
/// Empty `group_index` selects contiguous groups; otherwise it has `width` I32
/// entries shared by all rows. Callers must supply indices in `[0, groups)`.
#[derive(Clone, Debug)]
pub struct PackedTensor<R:Runtime> {
    pub codes:RudaTensor<R>, pub scales:RudaTensor<R>, pub offsets:RudaTensor<R>,
    pub group_index:RudaTensor<R>, pub shape:Shape,
    pub format:PackedFormat, pub group_size:usize,
}
impl<R:Runtime> PackedTensor<R> {
    pub fn new(codes:RudaTensor<R>,scales:RudaTensor<R>,offsets:RudaTensor<R>,
        group_index:RudaTensor<R>,shape:Shape,format:PackedFormat,group_size:usize)->Self {
        let this=Self {codes,scales,offsets,group_index,shape,format,group_size};
        this.validate(); this
    }
    pub fn width(&self)->usize { *self.shape.last().expect("packed tensor needs a feature axis") }
    pub fn rows(&self)->usize { self.shape.num_elements()/self.width() }
    pub fn groups(&self)->usize {
        if self.group_index.meta.num_elements()!=0 && self.scales.meta.num_dims()==2
            && self.scales.meta.shape()[0]==self.rows() {
            self.scales.meta.shape()[1]
        } else { self.width().div_ceil(self.group_size) }
    }
    pub fn row_bytes(&self)->usize { (self.width()*self.format.bits()).div_ceil(8) }
    pub fn validate(&self) {
        assert!(!self.shape.is_empty() && self.width()>0 && self.group_size>0);
        let elements=self.shape.iter().try_fold(1usize,|n,&d|n.checked_mul(d)).expect("packed shape overflow");
        assert!(elements<=u32::MAX as usize && self.width()<=u32::MAX as usize/8);
        assert_eq!(self.codes.dtype,DType::U8);
        assert_eq!(self.scales.dtype,DType::F32); assert_eq!(self.offsets.dtype,DType::F32);
        assert_eq!(self.group_index.dtype,DType::I32);
        assert_eq!(self.codes.meta.num_elements(),self.rows()*self.row_bytes());
        assert!(self.groups()>0 && self.groups()<=u32::MAX as usize);
        assert_eq!(self.scales.meta.num_elements(),self.rows()*self.groups());
        assert!(self.scales.meta.num_elements()<=u32::MAX as usize);
        assert_eq!(self.offsets.meta.num_elements(),self.scales.meta.num_elements());
        assert!(self.group_index.meta.num_elements()==0 || self.group_index.meta.num_elements()==self.width());
        for t in [&self.codes,&self.scales,&self.offsets,&self.group_index] { self.check(t); }
    }
    pub fn check(&self,t:&RudaTensor<R>) {
        assert!(t.is_contiguous() && t.qparams.is_none(),"packed operands require plain contiguous storage");
        assert_eq!(t.device.to_id(),self.codes.device.to_id());
        assert!(t.client.same_execution_queue(&self.codes.client),"packed operands must share a queue");
    }
    pub fn dequantize(&self,dtype:DType)->RudaTensor<R> {
        assert!(matches!(dtype,DType::F32|DType::F16|DType::BF16));
        let out=empty_device_contiguous_dtype(self.codes.client.clone(),self.codes.device.clone(),self.shape.clone(),dtype);
        if out.meta.num_elements()!=0 {
            decode::launch::<R>(&out.client,grid(out.meta.num_elements()),RudaDim::new_1d(128),
                self.codes.clone().into_array_arg(),self.scales.clone().into_array_arg(),
                self.offsets.clone().into_array_arg(),self.group_index.clone().into_array_arg(),out.clone().into_array_arg(),
                self.width(),self.group_size,self.groups(),self.format.code(),self.group_index.meta.num_elements()!=0,dtype.into());
        }
        out
    }
    /// Quantize with caller-supplied calibrated parameters. Zero scales encode a
    /// constant offset; floating formats saturate finite overflow during PTQ.
    pub fn quantize(input:&RudaTensor<R>,scales:RudaTensor<R>,offsets:RudaTensor<R>,
        format:PackedFormat,group_size:usize)->Self {
        assert!(matches!(input.dtype,DType::F32|DType::F16|DType::BF16));
        let shape=input.meta.shape().clone();
        let width=*shape.last().expect("quantize needs a feature axis");
        assert!(width>0 && width<=u32::MAX as usize/8 && group_size>0);
        let rows=shape.num_elements()/width;
        let codes=empty_device_contiguous_dtype(input.client.clone(),input.device.clone(),Shape::new([rows,(width*format.bits()).div_ceil(8)]),DType::U8);
        let index=empty_device_contiguous_dtype(input.client.clone(),input.device.clone(),Shape::new([0]),DType::I32);
        let result=Self::new(codes,scales,offsets,index,shape,format,group_size);
        result.check(input);
        if rows!=0 { encode::launch::<R>(&input.client,grid(result.codes.meta.num_elements()),RudaDim::new_1d(128),
            input.clone().into_array_arg(),result.scales.clone().into_array_arg(),result.offsets.clone().into_array_arg(),
            result.codes.clone().into_array_arg(),width,group_size,format.code(),input.dtype.into()); }
        result
    }
    /// Frozen-weight linear, or its input-gradient transpose. Decoding occurs
    /// inside the dot product, never in a full-sized floating weight allocation.
    pub fn linear(&self,x:&RudaTensor<R>,transpose:bool)->RudaTensor<R> {
        self.validate(); self.check(x);
        assert_eq!(self.shape.len(),2); assert_eq!(x.meta.num_dims(),2);
        assert!(matches!(x.dtype,DType::F32|DType::F16|DType::BF16));
        let (outputs,inputs)=(self.rows(),self.width());
        let (inner,width)=if transpose {(outputs,inputs)} else {(inputs,outputs)};
        assert_eq!(x.meta.shape()[1],inner);
        let size=x.meta.shape()[0].checked_mul(width).expect("packed linear output overflow");
        assert!(size<=u32::MAX as usize);
        let out=empty_device_contiguous_dtype(x.client.clone(),x.device.clone(),Shape::new([x.meta.shape()[0],width]),x.dtype);
        if size!=0 { linear::launch::<R>(&x.client,grid(size),RudaDim::new_1d(128),
            x.clone().into_array_arg(),self.codes.clone().into_array_arg(),self.scales.clone().into_array_arg(),
            self.offsets.clone().into_array_arg(),self.group_index.clone().into_array_arg(),out.clone().into_array_arg(),
            inputs,outputs,self.group_size,self.groups(),self.format.code(),transpose,self.group_index.meta.num_elements()!=0,x.dtype.into()); }
        out
    }
    pub fn linear_tensor_core(&self,x:&RudaTensor<R>,transpose:bool)->RudaTensor<R> {
        self.validate(); self.check(x);
        assert_eq!(self.shape.len(),2); assert_eq!(x.meta.num_dims(),2);
        assert!(matches!(x.dtype,DType::F16|DType::BF16),"packed Tensor Core input must be FP16/BF16");
        let (outputs,inputs)=(self.shape[0],self.shape[1]);
        let inner=if transpose {outputs} else {inputs};
        let columns=if transpose {inputs} else {outputs};
        assert_eq!(x.meta.shape()[1],inner);
        let rows=x.meta.shape()[0];
        let size=rows.checked_mul(columns).expect("packed Tensor Core shape overflow");
        assert!(size<=u32::MAX as usize);
        let cfg=MmaConfig {a_type:x.dtype.into(),b_type:x.dtype.into(),
            cd_type:ElemType::Float(FloatKind::F32).into(),m:16,n:16,k:16};
        let hardware=&x.client.properties().hardware;
        assert!(hardware.plane_size_min==32 && hardware.plane_size_max==32
            && hardware.max_shared_memory_size>=2048 && x.client.features().matmul.cmma.contains(&cfg)
            && columns.div_ceil(16)<=hardware.max_ruda_count.0 as usize
            && rows.div_ceil(16)<=hardware.max_ruda_count.1 as usize,
            "requested packed Tensor Core configuration is unavailable");
        let out=empty_device_contiguous_dtype(x.client.clone(),x.device.clone(),Shape::new([rows,columns]),x.dtype);
        if size!=0 { cooperative_linear::launch::<R>(&x.client,
            RudaCount::Static(columns.div_ceil(16) as u32,rows.div_ceil(16) as u32,1),RudaDim::new_1d(32),
            x.clone().into_array_arg(),self.codes.clone().into_array_arg(),self.scales.clone().into_array_arg(),
            self.offsets.clone().into_array_arg(),self.group_index.clone().into_array_arg(),out.clone().into_array_arg(),
            rows as u32,inputs,outputs,self.group_size,self.groups(),self.format.code(),transpose,
            self.group_index.meta.num_elements()!=0,x.dtype.into()); }
        out
    }
    pub fn prepare_linear(&self,x:&RudaTensor<R>,out:&RudaTensor<R>,transpose:bool,tensor_core:bool)->PreparedKernel<R> {
        self.validate();self.check(x);self.check(out);
        assert_eq!(self.shape.len(),2);assert_eq!(x.meta.num_dims(),2);
        assert_eq!(x.dtype,out.dtype);
        assert!(matches!(x.dtype,DType::F32|DType::F16|DType::BF16));
        let (outputs,inputs)=(self.rows(),self.width());let rows=x.meta.shape()[0];
        let (inner,columns)=if transpose {(outputs,inputs)} else {(inputs,outputs)};
        assert_eq!(x.meta.shape()[1],inner);assert_eq!(out.meta.shape().as_slice(),&[rows,columns]);
        assert!(out.meta.num_elements()>0 && out.meta.num_elements()<=u32::MAX as usize);
        if tensor_core {
            assert!(matches!(x.dtype,DType::F16|DType::BF16));
            let cfg=MmaConfig{a_type:x.dtype.into(),b_type:x.dtype.into(),cd_type:ElemType::Float(FloatKind::F32).into(),m:16,n:16,k:16};
            let h=&x.client.properties().hardware;
            assert!(h.plane_size_min==32 && h.plane_size_max==32 && h.max_shared_memory_size>=2048
                && x.client.features().matmul.cmma.contains(&cfg) && columns.div_ceil(16)<=h.max_ruda_count.0 as usize
                && rows.div_ceil(16)<=h.max_ruda_count.1 as usize,"packed graph Tensor Core configuration unavailable");
            unsafe {cooperative_linear::prepare::<R>(&x.client,RudaCount::Static(columns.div_ceil(16) as u32,rows.div_ceil(16) as u32,1),RudaDim::new_1d(32),
                x.clone().into_array_arg(),self.codes.clone().into_array_arg(),self.scales.clone().into_array_arg(),self.offsets.clone().into_array_arg(),
                self.group_index.clone().into_array_arg(),out.clone().into_array_arg(),rows as u32,inputs,outputs,self.group_size,self.groups(),self.format.code(),transpose,
                self.group_index.meta.num_elements()!=0,x.dtype.into())}
        } else {unsafe {linear::prepare::<R>(&x.client,grid(out.meta.num_elements()),RudaDim::new_1d(128),
            x.clone().into_array_arg(),self.codes.clone().into_array_arg(),self.scales.clone().into_array_arg(),self.offsets.clone().into_array_arg(),
            self.group_index.clone().into_array_arg(),out.clone().into_array_arg(),inputs,outputs,self.group_size,self.groups(),self.format.code(),transpose,
            self.group_index.meta.num_elements()!=0,x.dtype.into())}}
    }
    pub fn prepare_decode(&self,out:&RudaTensor<R>)->PreparedKernel<R> {
        self.validate();self.check(out);assert_eq!(self.shape.as_slice(),out.meta.shape().as_slice());
        assert!(matches!(out.dtype,DType::F32|DType::F16|DType::BF16) && out.meta.num_elements()>0);
        unsafe {decode::prepare::<R>(&out.client,grid(out.meta.num_elements()),RudaDim::new_1d(128),self.codes.clone().into_array_arg(),
            self.scales.clone().into_array_arg(),self.offsets.clone().into_array_arg(),self.group_index.clone().into_array_arg(),out.clone().into_array_arg(),
            self.width(),self.group_size,self.groups(),self.format.code(),self.group_index.meta.num_elements()!=0,out.dtype.into())}
    }
    /// Decode only selected embedding rows. Index bounds are caller-validated.
    pub fn embedding(&self,index:&RudaTensor<R>,dtype:DType)->RudaTensor<R> {
        self.validate();self.check(index);assert_eq!(self.shape.len(),2);
        assert_eq!(index.dtype,DType::I64);
        assert!(matches!(dtype,DType::F32|DType::F16|DType::BF16));
        let shape=Shape::new([index.meta.num_elements(),self.width()]);
        assert!(shape.num_elements()<=u32::MAX as usize);
        let out=empty_device_contiguous_dtype(index.client.clone(),index.device.clone(),shape,dtype);
        if out.meta.num_elements()!=0 {embedding::launch::<R>(&out.client,grid(out.meta.num_elements()),RudaDim::new_1d(128),
            index.clone().into_array_arg(),self.codes.clone().into_array_arg(),self.scales.clone().into_array_arg(),
            self.offsets.clone().into_array_arg(),self.group_index.clone().into_array_arg(),out.clone().into_array_arg(),
            self.width(),self.group_size,self.groups(),self.format.code(),self.group_index.meta.num_elements()!=0,dtype.into());}
        out
    }
}
fn grid(n:usize)->RudaCount { RudaCount::Static(u32::try_from(n.div_ceil(128)).expect("packed grid overflow"),1,1) }

/// Stable ascending row sort for exact percentile calibration. Two bounded
/// device work buffers are reused across merge passes; no value readback.
pub fn sort_rows<R:Runtime>(x:&RudaTensor<R>)->RudaTensor<R> {
    assert_eq!(x.dtype,DType::F32);assert_eq!(x.meta.num_dims(),2);assert!(x.is_contiguous());
    let width=x.meta.shape()[1];assert!(width>0 && x.meta.num_elements()<=u32::MAX as usize);
    if width==1 || x.meta.num_elements()==0 {return x.clone();}
    let a=empty_device_contiguous_dtype(x.client.clone(),x.device.clone(),x.meta.shape().clone(),DType::F32);
    let b=empty_device_contiguous_dtype(x.client.clone(),x.device.clone(),x.meta.shape().clone(),DType::F32);
    let mut source=x.clone();let mut run=1usize;let mut first=true;
    while run<width {
        let out=if first {&a} else {&b};
        merge_rows::launch::<R>(&x.client,grid(x.meta.num_elements()),RudaDim::new_1d(128),
            source.into_array_arg(),out.clone().into_array_arg(),width as u32,run as u32);
        source=out.clone();first=!first;run*=2;
    }
    source
}

#[ruda(launch)]
fn merge_rows(x:&Array<f32>,out:&mut Array<f32>,width:u32,run:u32) {
    let i=ABSOLUTE_POS as usize;
    if i<x.len() {
        let width=width as usize;let run=run as usize;
        let row=i/width*width;let col=i%width;let begin=col/(run*2)*(run*2);
        let middle=usize::min(begin+run,width);let end=usize::min(middle+run,width);
        let left=col<middle;let mut own=begin;let mut other=middle;let mut low=middle;let mut high=end;
        if !left {own=middle;other=begin;low=begin;high=middle;}
        let value=x[i];
        while low<high {
            let mid=low+(high-low)/2;let candidate=x[row+mid];
            let mut advance=candidate<value;
            if !left {advance=candidate<=value;}
            if advance {low=mid+1;} else {high=mid;}
        }
        out[row+begin+(col-own)+(low-other)]=value;
    }
}

#[ruda]
pub fn bits(#[comptime] format:u32)->usize {
    usize::cast_from(comptime!(if format==14 {4usize} else if format>=18 {8usize} else {format as usize}))
}

/// Software minifloat decode, usable on hardware without native FP8/FP4 casts.
#[ruda]
pub fn decode_code(code:u32,#[comptime] format:u32)->f32 {
    if comptime!(format<14) { code as f32 }
    else if comptime!(format==14) {
        let m=code&7;
        let mut value=m as f32*0.5;
        if m>=4 { value=(m-2) as f32; }
        if m==7 { value=6.0; }
        if code&8!=0 { value=-value; }
        value
    } else {
        let mb=comptime!(if format==18 {3u32} else {2u32});
        let bias=comptime!(if format==18 {7i32} else {15i32});
        let sign=(code&128)<<24;
        let exponent=(code&127)>>mb;
        let mantissa=code&((1u32<<mb)-1);
        let mut value=0.0f32;
        if exponent==0 {
            let unit=f32::from_bits(((128-bias-mb as i32) as u32)<<23);
            value=mantissa as f32*unit;
        } else {
            value=f32::from_bits((((exponent as i32-bias+127) as u32)<<23)|(mantissa<<(23-mb)));
        }
        if comptime!(format==18) { if (code&127)==127 { value=f32::from_bits(0x7fc00000u32); } }
        else { if exponent==31 { value=f32::from_bits(0x7f800000u32|(mantissa<<21)); } }
        f32::from_bits(value.to_bits()|sign)
    }
}

#[ruda]
pub fn encode_code(value:f32,#[comptime] format:u32,#[comptime] saturate:bool)->u32 {
    if comptime!(format<14) {
        let maximum=((1u32<<format)-1) as f32;
        let clipped=f32::min(f32::max(value,0.0),maximum);
        let lower=clipped.floor();
        let mut code=lower as u32;
        if clipped-lower>0.5 || (clipped-lower==0.5 && code&1!=0) { code+=1; }
        code
    } else {
        let negative=value.to_bits()>>31;
        let sign_bit=comptime!(if format==14 {8u32} else {128u32});
        let last=comptime!(if format==14 {7u32} else if format==18 {126u32} else {123u32});
        let magnitude=value.abs();
        let mut low=0u32; let mut high=u32::cast_from(last);
        while low<high {
            let mid=(low+high)/2;
            if decode_code(mid,format)<magnitude { low=mid+1; } else { high=mid; }
        }
        let mut code=low;
        if low>0 {
            let distance=magnitude-decode_code(low-1,format);
            let upper=decode_code(low,format)-magnitude;
            if distance<upper || (distance==upper && low&1!=0) { code=low-1; }
        }
        if comptime!(!saturate && format!=14) {
            let overflow=comptime!(if format==18 {464.0f32} else {61440.0f32});
            if magnitude>=overflow {
                code=comptime!(if format==18 {127u32} else {124u32});
                // E4M3's 464 tie rounds to the finite even code 126.
                if comptime!(format==18) { if magnitude==overflow { code=126; } }
            }
        }
        if value!=value { code=comptime!(if format==14 {0u32} else {127u32}); }
        code|negative*sign_bit
    }
}

#[ruda]
pub fn load(codes:&Array<u8>,scales:&Array<f32>,offsets:&Array<f32>,indices:&Array<i32>,
    row:usize,col:usize,#[comptime] width:usize,#[comptime] group:usize,
    #[comptime] format:u32,#[comptime] indexed:bool)->f32 {
    load_grouped(codes,scales,offsets,indices,row,col,width,group,(width+group-1)/group,format,indexed)
}

#[ruda]
pub fn load_grouped(codes:&Array<u8>,scales:&Array<f32>,offsets:&Array<f32>,indices:&Array<i32>,
    row:usize,col:usize,#[comptime] width:usize,#[comptime] group:usize,
    #[comptime] parameter_groups:usize,#[comptime] format:u32,#[comptime] indexed:bool)->f32 {
    let b=bits(format); let stride=(width*b+7)/8;
    let bit=col*b; let at=row*stride+bit/8;
    let mut word=codes[at] as u32;
    if bit%8+b>8 { word|=(codes[at+1] as u32)<<8; }
    let code=(word>>(bit%8) as u32)&((1u32<<(b as u32))-1);
    let mut g=col/group;
    if comptime!(indexed) { g=indices[col] as usize; }
    let param=row*parameter_groups+g;
    // Separate operations preserve imported scale/offset semantics.
    decode_code(code,format)*scales[param]+offsets[param]
}

#[ruda(launch)]
fn decode<F:Float>(codes:&Array<u8>,scales:&Array<f32>,offsets:&Array<f32>,indices:&Array<i32>,
    out:&mut Array<F>,#[comptime] width:usize,#[comptime] group:usize,
    #[comptime] parameter_groups:usize,#[comptime] format:u32,#[comptime] indexed:bool,#[define(F)] _dtype:StorageType) {
    let i=ABSOLUTE_POS as usize;
    if i<out.len() { out[i]=F::cast_from(load_grouped(codes,scales,offsets,indices,i/width,i%width,width,group,parameter_groups,format,indexed)); }
}

#[ruda(launch)]
fn encode<F:Float>(x:&Array<F>,scales:&Array<f32>,offsets:&Array<f32>,out:&mut Array<u8>,
    #[comptime] width:usize,#[comptime] group:usize,#[comptime] format:u32,#[define(F)] _dtype:StorageType) {
    let i=ABSOLUTE_POS as usize;
    if i<out.len() {
        let b=bits(format); let stride=(width*b+7)/8;
        let row=i/stride; let byte=i%stride; let first=(byte*8)/b;
        let last=usize::min(width,((byte+1)*8+b-1)/b);
        let mut word=0u32;
        for col in first..last {
            let p=row*((width+group-1)/group)+col/group;
            let mut value=0.0f32;
            if scales[p]!=0.0 { value=(f32::cast_from(x[row*width+col])-offsets[p])/scales[p]; }
            let code=encode_code(value,format,true);
            if col*b>=byte*8 { word|=code<<((col*b-byte*8) as u32); }
            else { word|=code>>((byte*8-col*b) as u32); }
        }
        out[i]=(word&255) as u8;
    }
}

#[ruda(launch)]
fn linear<F:Float>(x:&Array<F>,codes:&Array<u8>,scales:&Array<f32>,offsets:&Array<f32>,indices:&Array<i32>,
    out:&mut Array<F>,#[comptime] inputs:usize,#[comptime] outputs:usize,#[comptime] group:usize,
    #[comptime] parameter_groups:usize,#[comptime] format:u32,#[comptime] transpose:bool,#[comptime] indexed:bool,#[define(F)] _dtype:StorageType) {
    let i=ABSOLUTE_POS as usize;
    if i<out.len() {
        let inner=comptime!(if transpose {outputs} else {inputs});
        let width=comptime!(if transpose {inputs} else {outputs});
        let batch=i/width; let col=i%width;
        let mut sum=0.0f32;
        for j in 0usize..inner {
            let mut wr=col; let mut wc=j;
            if comptime!(transpose) { wr=j; wc=col; }
            sum=fma(f32::cast_from(x[batch*inner+j]),load_grouped(codes,scales,offsets,indices,wr,wc,inputs,group,parameter_groups,format,indexed),sum);
        }
        out[i]=F::cast_from(sum);
    }
}

#[ruda(launch)]
fn cooperative_linear<F:Float>(x:&Array<F>,codes:&Array<u8>,scales:&Array<f32>,offsets:&Array<f32>,indices:&Array<i32>,
    out:&mut Array<F>,rows:u32,#[comptime] inputs:usize,#[comptime] outputs:usize,
    #[comptime] group:usize,#[comptime] parameter_groups:usize,#[comptime] format:u32,#[comptime] transpose:bool,
    #[comptime] indexed:bool,#[define(F)] _dtype:StorageType) {
    let row_base=RUDA_POS_Y as usize*16;
    let col_base=RUDA_POS_X as usize*16;
    let lane=UNIT_POS as usize;
    let inner=comptime!(if transpose {outputs} else {inputs});
    let columns=comptime!(if transpose {inputs} else {outputs});
    let mut left=SharedMemory::<F>::new_aligned(256usize,32usize);
    let mut right=SharedMemory::<F>::new_aligned(256usize,32usize);
    let mut result=SharedMemory::<f32>::new_aligned(256usize,32usize);
    let acc=cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,
        16usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0);
    let mut base=0usize;
    while base<inner {
        #[unroll]
        for i in 0usize..8usize {
            let t=lane+i*32;
            let row=row_base+t/16;
            let col=col_base+t%16;
            let ka=base+t%16;
            let kb=base+t/16;
            let mut a=F::cast_from(0.0f32);
            let mut b=F::cast_from(0.0f32);
            if row<rows as usize && ka<inner {a=x[row*inner+ka];}
            if col<columns && kb<inner {
                let mut wr=col;let mut wc=kb;
                if comptime!(transpose) {wr=kb;wc=col;}
                b=F::cast_from(load_grouped(codes,scales,offsets,indices,wr,wc,inputs,group,parameter_groups,format,indexed));
            }
            left[t]=a;right[t]=b;
        }
        sync_ruda();
        let a=cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::A,
            16usize,16usize,16usize,cmma::MatrixLayout::RowMajor,&left.to_slice(),16);
        let b=cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::B,
            16usize,16usize,16usize,cmma::MatrixLayout::RowMajor,&right.to_slice(),16);
        cmma::execute::<F,F,f32,f32>(&a,&b,&acc,&acc);
        sync_ruda();
        if inner-base<=16 {base=inner;} else {base+=16;}
    }
    cmma::store(&mut result.to_slice_mut(),&acc,16,cmma::MatrixLayout::RowMajor);
    sync_ruda();
    #[unroll]
    for i in 0usize..8usize {
        let t=lane+i*32;
        let row=row_base+t/16;let col=col_base+t%16;
        if row<rows as usize && col<columns {out[row*columns+col]=F::cast_from(result[t]);}
    }
}

#[ruda(launch)]
fn embedding<F:Float>(index:&Array<i64>,codes:&Array<u8>,scales:&Array<f32>,offsets:&Array<f32>,groups:&Array<i32>,
    out:&mut Array<F>,#[comptime] width:usize,#[comptime] group:usize,#[comptime] parameter_groups:usize,#[comptime] format:u32,
    #[comptime] indexed:bool,#[define(F)] _dtype:StorageType) {
    let i=ABSOLUTE_POS as usize;
    if i<out.len() {out[i]=F::cast_from(load_grouped(codes,scales,offsets,groups,index[i/width] as usize,i%width,width,group,parameter_groups,format,indexed));}
}
