// ATen segment-reduction formulas; retained PyTorch license: bessel.rs.
use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn combine<S:Float,F:Float>(a:S,b:S,#[comptime] kind:u32)->S {
    let x=F::cast_from(a);let y=F::cast_from(b);let mut value=x;
    if comptime!(kind==0) {if y!=y || y>x {value=y;}}
    else if comptime!(kind==2) {if y!=y || y<x {value=y;}}
    else if comptime!(kind==4) {value=x*y;}else{value=x+y;}
    S::cast_from(value)
}

#[ruda]
fn identity<S:Float>(#[comptime] kind:u32)->S {
    let mut value=S::new(0.0);
    if comptime!(kind==0) {value=S::new(f32::NEG_INFINITY);}
    else if comptime!(kind==2) {value=S::new(f32::INFINITY);}
    else if comptime!(kind==4) {value=S::new(1.0);}
    value
}

#[ruda]
fn limits<I:Numeric+RudaElement>(offsets:&Tensor<I>,index:usize,width:usize)->(usize,usize) {
    let start=i64::cast_from(offsets[address(offsets,index)]);let end=i64::cast_from(offsets[address(offsets,index+1)]);
    if start<0 || end<start || end>i64::cast_from(width) {
        intrinsic!(|scope| {scope.register(ruda_core::ir::Branch::Unreachable);});
    }
    (usize::cast_from(start),usize::cast_from(end))
}

#[ruda]
fn divide_count<S:Float,F:Float>(value:S,count:usize)->S {
    S::cast_from(F::cast_from(value)/F::cast_from(S::cast_from(count)))
}

#[ruda(launch)]
fn forward<S:Float+RudaElement,F:Float+RudaElement,I:Numeric+RudaElement>(input:&Tensor<S>,offsets:&Tensor<I>,out:&mut Tensor<S>,initial:F,#[comptime] kind:u32,#[comptime] has_initial:bool) {
    let index=ABSOLUTE_POS as usize;
    if index<out.len() {
        let inner=input.shape(2);let segments=out.shape(1);let row=index/inner;let outer=row/segments;let feature=index%inner;
        let (start,end)=limits::<I>(offsets,outer*(segments+1)+row%segments,input.shape(1));
        let mut value=S::cast_from(initial);
        for position in start..end {value=combine::<S,F>(value,input[address(input,(outer*input.shape(1)+position)*inner+feature)],kind);}
        if comptime!(kind==1) {
            if end==start {if comptime!(!has_initial) {value=S::new(f32::NAN);}}
            else if value==value {value=divide_count::<S,F>(value,end-start);}
        }
        out[address(out,index)]=value;
    }
}

#[ruda(launch)]
fn forward_vector<S:Float+RudaElement,F:Float+RudaElement,I:Numeric+RudaElement>(input:&Tensor<S>,offsets:&Tensor<I>,out:&mut Tensor<S>,initial:F,#[comptime] kind:u32,#[comptime] has_initial:bool) {
    let segment=(RUDA_POS_X+RUDA_POS_Y*RUDA_COUNT_X) as usize;let lane=UNIT_POS as usize;
    if segment<out.shape(1) {
        let (start,end)=limits::<I>(offsets,segment,input.shape(1));let mut value=identity::<S>(kind);
        let mut position=i64::cast_from(start)+i64::cast_from(lane);
        while position<i64::cast_from(end) {value=combine::<S,F>(value,input[address(input,usize::cast_from(position))],kind);position+=256;}
        let mut shared=SharedMemory::<S>::new(256usize);shared[lane]=value;sync_ruda();let mut step=128usize;
        while step>0 {
            if lane<step {shared[lane]=combine::<S,F>(shared[lane],shared[lane+step],kind);}
            sync_ruda();step/=2;
        }
        if lane==0 {
            value=S::cast_from(initial);
            if end>start {value=combine::<S,F>(value,shared[0],kind);}
            if comptime!(kind==1) {
                if end==start {if comptime!(!has_initial) {value=S::new(f32::NAN);}}
                else if value==value {value=divide_count::<S,F>(value,end-start);}
            }
            out[address(out,segment)]=value;
        }
    }
}

#[ruda(launch)]
fn backward<S:Float+RudaElement,F:Float+RudaElement,I:Numeric+RudaElement>(input:&Tensor<S>,offsets:&Tensor<I>,output:&Tensor<S>,grad:&Tensor<S>,out:&mut Tensor<S>,initial:F,#[comptime] kind:u32) {
    let index=ABSOLUTE_POS as usize;
    if index<output.len() {
        let inner=input.shape(2);let segments=output.shape(1);let row=index/inner;let outer=row/segments;let feature=index%inner;
        let (start,end)=limits::<I>(offsets,outer*(segments+1)+row%segments,input.shape(1));
        let g=grad[address(grad,index)];let y=output[address(output,index)];
        if comptime!(kind==0 || kind==2) {
            let mut count=0usize;
            for position in start..end {
                let at=(outer*input.shape(1)+position)*inner+feature;let x=input[address(input,at)];
                if x!=x || x==y {out[address(out,at)]=g;count+=1;}
            }
            if count>1 {
                for position in start..end {
                    let at=address(out,(outer*input.shape(1)+position)*inner+feature);let value=out[at];
                    if value>S::new(0.0) {out[at]=divide_count::<S,F>(value,count);}
                }
            }
        } else if comptime!(kind==1 || kind==3) {
            if end>start {
                let mut value=g;if comptime!(kind==1) {value=divide_count::<S,F>(g,end-start);}
                for position in start..end {out[address(out,(outer*input.shape(1)+position)*inner+feature)]=value;}
            }
        } else {
            let scaled=S::cast_from(F::cast_from(g)*F::cast_from(y));
            for position in start..end {
                let at=(outer*input.shape(1)+position)*inner+feature;let x=input[address(input,at)];let mut value=S::new(0.0);
                if x!=x || x==S::new(0.0) {
                    let mut product=S::cast_from(initial);
                    for other in start..end {
                        if other!=position {product=combine::<S,F>(product,input[address(input,(outer*input.shape(1)+other)*inner+feature)],4u32);}
                    }
                    value=S::cast_from(F::cast_from(g)*F::cast_from(product));
                } else {value=S::cast_from(F::cast_from(scaled)/F::cast_from(x));}
                out[address(out,at)]=value;
            }
        }
    }
}

pub(super) fn dispatch(op:u32,xs:&[View],ints:&[i64],values:&[f64]) {
    assert!(matches!(op,366|367));assert_eq!(xs.len(),if op==366 {3}else{5});assert_eq!(ints.len(),3);assert_eq!(values.len(),1);
    assert!((0..=4).contains(&ints[0]));assert!(ints[1..].iter().all(|&n|matches!(n,0|1)));
    let input=&xs[0];let offsets=&xs[1];let output=&xs[2];assert_eq!(input.shape.len(),3);assert_eq!(output.shape.len(),3);
    assert_eq!(offsets.shape,vec![input.shape[0],output.shape[1]+1]);assert_eq!(output.shape[0],input.shape[0]);assert_eq!(output.shape[2],input.shape[2]);
    assert!(matches!(input.dtype,0|1|2|9));assert!(matches!(offsets.dtype,4|5));for value in &xs[2..] {assert_eq!(value.dtype,input.dtype);}
    if op==367 {assert_eq!(xs[3].shape,output.shape);assert_eq!(xs[4].shape,input.shape);}
    if output.len==0 {return;}
    let parallel=op==366 && ints[2]!=0;let kind=ints[0] as u32;let has_initial=ints[1]!=0;let c=client();let block=RudaDim::new_1d(256);
    let grid;
    if parallel {
        assert_eq!(input.shape[0],1);assert_eq!(input.shape[2],1);let count=u32::try_from(output.shape[1]).expect("segment count overflow");let x=count.min(1<<20);
        grid=RudaCount::Static(x,count.div_ceil(x),1);
    } else {grid=RudaCount::Static(u32::try_from(output.len.div_ceil(256)).expect("segment launch overflow"),1,1);}
    macro_rules! run {($s:ty,$f:ty,$i:ty)=>{unsafe {
        if op==367 {backward::launch::<$s,$f,$i,CudaRuntime>(&c,grid,block,input.arg(),offsets.arg(),output.arg(),xs[3].arg(),xs[4].arg(),values[0] as $f,kind);}
        else if parallel {forward_vector::launch::<$s,$f,$i,CudaRuntime>(&c,grid,block,input.arg(),offsets.arg(),output.arg(),values[0] as $f,kind,has_initial);}
        else {forward::launch::<$s,$f,$i,CudaRuntime>(&c,grid,block,input.arg(),offsets.arg(),output.arg(),values[0] as $f,kind,has_initial);}
    }};}
    macro_rules! index {($s:ty,$f:ty)=>{if offsets.dtype==4 {run!($s,$f,i64);}else{run!($s,$f,i32);}};}
    match input.dtype {0=>index!(f32,f32),1=>index!(f16,f32),2=>index!(bf16,f32),9=>index!(f64,f64),_=>unreachable!()}
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);
}
