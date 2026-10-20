use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn wrap_indices<I:Numeric+RudaElement>(indices:&Tensor<I>,out:&mut Tensor<I>,rows:i64) {
    let position=ABSOLUTE_POS as usize;
    if position<indices.len() {
        let mut row=i64::cast_from(indices[address(indices,position)]);
        if row < -rows || row>=rows {
            intrinsic!(|scope| {scope.register(ruda_core::ir::Branch::Unreachable);});
        }
        if row<0 {row+=rows;}
        out[address(out,position)]=I::cast_from(row);
    }
}

#[ruda]
fn warp_sum<F:Float+RudaElement>(input:F)->F {
    let mut value=input;let mut offset=16u32;
    while offset>0 {value+=plane_shuffle_down(value,offset);offset/=2;}
    value
}

#[ruda(launch)]
fn embedding_renorm<S:Float+RudaElement,F:Float+RudaElement,I:Numeric+RudaElement>(weight:&mut Tensor<S>,indices:&Tensor<I>,maximum:F,p:F) {
    let position=RUDA_POS_X as usize;let lane=UNIT_POS as usize;
    let row=i64::cast_from(indices[address(indices,position)]);let mut unique=true;
    if position!=0 {unique=indices[address(indices,position-1)]!=indices[address(indices,position)];}
    if unique {
        if row<0 || row>=i64::cast_from(weight.shape(0)) {
            intrinsic!(|scope| {scope.register(ruda_core::ir::Branch::Unreachable);});
        }
        let width=i64::cast_from(weight.shape(1));let base=usize::cast_from(row)*weight.stride(0);
        let mut shared=SharedMemory::<F>::new(4usize);let mut value=F::new(0.0);let mut feature=i64::cast_from(lane);
        while feature<width {
            let x=F::cast_from(weight[base+usize::cast_from(feature)*weight.stride(1)]);
            if p==F::new(1.0) {value+=x.abs();}
            else if p==F::new(2.0) {value+=x*x;}
            else {value+=x.powf(p);}
            feature+=128;
        }
        value=warp_sum::<F>(value);sync_ruda();
        if lane%32==0 {shared[lane/32]=value;}
        sync_ruda();value=F::new(0.0);
        if lane<4 {value=shared[lane];}
        if lane<32 {value=warp_sum::<F>(value);}
        if lane==0 {shared[0]=value.powf(F::cast_from(1.0f64/f64::cast_from(p)));}
        sync_ruda();let norm=shared[0];
        if norm>maximum {
            let factor=S::cast_from(f64::cast_from(maximum)/(f64::cast_from(norm)+1e-7f64));
            feature=i64::cast_from(lane);
            while feature<width {
                let at=base+usize::cast_from(feature)*weight.stride(1);weight[at]=weight[at]*factor;feature+=128;
            }
        }
    }
}

#[ruda(launch)]
fn renorm_factor<F:Float+RudaElement>(norms:&Tensor<F>,out:&mut Tensor<F>,maximum:F) {
    let index=ABSOLUTE_POS as usize;
    if index<out.len() {
        let norm=norms[address(norms,index)];let mut value=F::new(1.0);
        if norm>maximum {value=maximum/(norm+F::cast_from(1e-7f64));}
        out[address(out,index)]=value;
    }
}

#[ruda(launch)]
fn repeat_indices<I:Numeric+RudaElement>(repeats:&Tensor<I>,prefix:&Tensor<i64>,out:&mut Tensor<I>) {
    let lane=UNIT_POS as usize%32;
    let mut sample=i64::cast_from(RUDA_POS_X)*16+i64::cast_from(UNIT_POS/32);
    let stride=i64::cast_from(RUDA_COUNT_X)*16;
    let size=i64::cast_from(repeats.len());let total=i64::cast_from(out.len());
    if prefix[address(prefix,repeats.len()-1)]!=total {
        intrinsic!(|scope| {scope.register(ruda_core::ir::Branch::Unreachable);});
    }
    while sample<size {
        let count=i64::cast_from(repeats[address(repeats,usize::cast_from(sample))]);
        let end=prefix[address(prefix,usize::cast_from(sample))];let start=end-count;
        if count<0 || start<0 || end>total {
            intrinsic!(|scope| {scope.register(ruda_core::ir::Branch::Unreachable);});
        }
        let mut output_index=start+i64::cast_from(lane);
        while output_index<end {
            out[address(out,usize::cast_from(output_index))]=I::cast_from(sample);output_index+=32;
        }
        sample+=stride;
    }
}

#[ruda(launch)]
fn bincount<I:Numeric+RudaElement,S:Numeric+RudaElement>(input:&Tensor<I>,weights:&Tensor<S>,out:&mut Tensor<Atomic<S>>,#[comptime] weighted:bool) {
    let index=ABSOLUTE_POS as usize;
    if index<input.len() {
        let bin=i64::cast_from(input[address(input,index)]);
        if bin<0 || bin>=i64::cast_from(out.len()) {
            intrinsic!(|scope| {scope.register(ruda_core::ir::Branch::Unreachable);});
        }
        let mut value=S::cast_from(1);
        if comptime!(weighted) {value=weights[address(weights,index)];}
        out[usize::cast_from(bin)*out.stride(0)].fetch_add(value);
    }
}

fn grid(n:usize)->RudaCount {RudaCount::Static(u32::try_from(n.div_ceil(256)).expect("indexed operation launch overflow"),1,1)}

pub(super) fn dispatch(op:u32,xs:&[View],ints:&[i64],values:&[f64]) {
    assert!((345..=349).contains(&op));let c=client();let block=RudaDim::new_1d(256);
    if op==345 {
        assert_eq!(xs.len(),2);assert_eq!(ints.len(),1);assert!(ints[0]>=0);assert!(values.is_empty());
        assert_eq!(xs[0].dtype,xs[1].dtype);assert!(matches!(xs[0].dtype,4|5));assert_eq!(xs[0].len,xs[1].len);
        if xs[0].len==0 {return;}
        macro_rules! run {($i:ty)=>{unsafe {wrap_indices::launch::<$i,CudaRuntime>(&c,grid(xs[0].len),block,xs[0].arg(),xs[1].arg(),ints[0]);}};}
        if xs[0].dtype==4 {run!(i64);}else{run!(i32);}
    } else if op==346 {
        assert_eq!(xs.len(),2);assert!(ints.is_empty());assert_eq!(values.len(),2);
        assert_eq!(xs[0].shape.len(),2);assert_eq!(xs[1].shape.len(),1);
        assert!(matches!(xs[0].dtype,0|1|2|9));assert!(matches!(xs[1].dtype,4|5));
        if xs[1].len==0 {return;}
        let launch=RudaCount::Static(u32::try_from(xs[1].len).unwrap(),1,1);let threads=RudaDim::new_1d(128);
        macro_rules! run {($s:ty,$f:ty,$i:ty)=>{unsafe {embedding_renorm::launch::<$s,$f,$i,CudaRuntime>(&c,launch,threads,xs[0].arg(),xs[1].arg(),values[0] as $f,values[1] as $f);}};}
        macro_rules! indexed {($s:ty,$f:ty)=>{if xs[1].dtype==4 {run!($s,$f,i64);}else{run!($s,$f,i32);}};}
        match xs[0].dtype {0=>indexed!(f32,f32),1=>indexed!(f16,f32),2=>indexed!(bf16,f32),9=>indexed!(f64,f64),_=>unreachable!()}
    } else if op==347 {
        assert_eq!(xs.len(),2);assert!(ints.is_empty());assert_eq!(values.len(),1);
        assert_eq!(xs[0].shape,xs[1].shape);assert_eq!(xs[0].dtype,xs[1].dtype);assert!(matches!(xs[0].dtype,0|9));
        if xs[0].len==0 {return;}
        macro_rules! run {($f:ty)=>{unsafe {renorm_factor::launch::<$f,CudaRuntime>(&c,grid(xs[0].len),block,xs[0].arg(),xs[1].arg(),values[0] as $f);}};}
        if xs[0].dtype==0 {run!(f32);}else{run!(f64);}
    } else if op==348 {
        assert_eq!(xs.len(),3);assert!(ints.is_empty());assert!(values.is_empty());
        for view in xs {assert_eq!(view.shape.len(),1);}
        assert_eq!(xs[0].len,xs[1].len);assert_eq!(xs[1].dtype,4);assert_eq!(xs[0].dtype,xs[2].dtype);assert!(matches!(xs[0].dtype,4|5));
        if xs[0].len==0 {return;}
        let launch=RudaCount::Static(u32::try_from(xs[0].len.div_ceil(16).min(2048)).unwrap(),1,1);
        macro_rules! run {($i:ty)=>{unsafe {repeat_indices::launch::<$i,CudaRuntime>(&c,launch,RudaDim::new_1d(512),xs[0].arg(),xs[1].arg(),xs[2].arg());}};}
        if xs[0].dtype==4 {run!(i64);}else{run!(i32);}
    } else {
        assert_eq!(xs.len(),3);assert_eq!(ints.len(),1);assert!(values.is_empty());assert!(matches!(ints[0],0|1));let weighted=ints[0]!=0;
        assert_eq!(xs[0].shape.len(),1);assert_eq!(xs[2].shape.len(),1);assert!((4..=8).contains(&xs[0].dtype));
        if weighted {assert_eq!(xs[1].len,xs[0].len);assert!(matches!(xs[2].dtype,0|9));}else{assert_eq!(xs[2].dtype,4);}
        assert_eq!(xs[1].dtype,xs[2].dtype);if xs[0].len==0 {return;}
        macro_rules! run {($i:ty,$s:ty)=>{unsafe {bincount::launch::<$i,$s,CudaRuntime>(&c,grid(xs[0].len),block,xs[0].arg(),xs[1].arg(),xs[2].arg(),weighted);}};}
        macro_rules! scalar {($i:ty)=>{match xs[2].dtype {0=>run!($i,f32),4=>run!($i,i64),9=>run!($i,f64),_=>unreachable!()}};}
        match xs[0].dtype {4=>scalar!(i64),5=>scalar!(i32),6=>scalar!(i16),7=>scalar!(i8),8=>scalar!(u8),_=>unreachable!()}
    }
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);
}
