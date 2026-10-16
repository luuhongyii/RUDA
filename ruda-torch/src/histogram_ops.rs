use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn histc<I:Numeric+RudaElement,B:Numeric+RudaElement,O:Numeric+RudaElement>(input:&Tensor<I>,out:&mut Tensor<Atomic<O>>,minimum:B,maximum:B) {
    let index=ABSOLUTE_POS as usize;
    if index<input.len() {
        let value=B::cast_from(input[address(input,index)]);
        if value>=minimum && value<=maximum {
            let mut bin=i64::cast_from((value-minimum)*B::cast_from(out.len())/(maximum-minimum));
            if bin==i64::cast_from(out.len()) {bin-=1;}
            if bin>=0 && bin<i64::cast_from(out.len()) {out[usize::cast_from(bin)].fetch_add(O::cast_from(1));}
        }
    }
}

#[ruda]
fn upper_bound<S:Float+RudaElement>(edges:&Tensor<S>,base:usize,value:S,begin:usize,end:usize)->usize {
    let mut low=begin;let mut high=end;
    while low<high {
        let middle=low+(high-low)/2;
        if value<edges[address(edges,base+middle)] {high=middle;}else{low=middle+1;}
    }
    low
}

#[ruda(launch)]
fn classify<S:Float+RudaElement>(input:&Tensor<S>,edges:&Tensor<S>,keys:&mut Tensor<i64>,#[comptime] bins:Vec<u32>,#[comptime] linear:bool) {
    let sample=ABSOLUTE_POS as usize;
    if sample<keys.len() {
        let counts=Array::<u32>::from_data(bins);let dimensions=input.shape(1);
        let mut edge_base=0usize;let mut target=0i64;let mut valid=true;
        for axis in 0..dimensions {
            let count=counts[axis] as usize;let value=input[address(input,sample*dimensions+axis)];
            let left=edges[address(edges,edge_base)];let right=edges[address(edges,edge_base+count)];
            if !(value>=left && value<=right) {valid=false;}
            if valid {
                let mut first=0usize;let mut last=count+1;
                if comptime!(linear) {
                    let delta=S::cast_from(value-left);let product=S::cast_from(delta*S::cast_from(count));
                    let width=S::cast_from(right-left);let estimate=S::cast_from(product/width);
                    if estimate>=S::new(0.0) && estimate<S::new(f32::INFINITY) {
                        let position=i64::cast_from(estimate);
                        if position<=i64::cast_from(count) {
                            first=usize::cast_from(i64::max(position-1,0));
                            last=usize::cast_from(i64::min(position+2,i64::cast_from(count+1)));
                        }
                    }
                }
                let mut position=i64::cast_from(upper_bound(edges,edge_base,value,first,last))-1;
                if position==i64::cast_from(count) {position-=1;}
                if position<0 || position>=i64::cast_from(count) {valid=false;}
                else {target=target*i64::cast_from(count)+position;}
            }
            edge_base+=count+1;
        }
        if !valid {target=-1;}
        keys[address(keys,sample)]=target;
    }
}

#[ruda(launch)]
fn accumulate<S:Float+RudaElement>(keys:&Tensor<i64>,weights:&Tensor<S>,hist:&mut Tensor<Atomic<S>>,#[comptime] weighted:bool) {
    let sample=ABSOLUTE_POS as usize;
    if sample<keys.len() {
        let bin=keys[address(keys,sample)];
        if bin>=0 {
            if bin>=i64::cast_from(hist.len()) {
                intrinsic!(|scope| {scope.register(ruda_core::ir::Branch::Unreachable);});
            }
            let mut value=S::new(1.0);
            if comptime!(weighted) {value=weights[address(weights,sample)];}
            hist[usize::cast_from(bin)].fetch_add(value);
        }
    }
}

#[ruda]
fn key_bound(keys:&Tensor<i64>,bin:i64,#[comptime] upper:bool)->usize {
    let mut low=0usize;let mut high=keys.len();
    while low<high {
        let middle=low+(high-low)/2;let key=keys[address(keys,middle)];
        let mut advance=key<bin;if comptime!(upper) {advance=key<=bin;}
        if advance {low=middle+1;}else{high=middle;}
    }
    low
}

#[ruda(launch)]
fn accumulate_ordered<S:Float+RudaElement>(keys:&Tensor<i64>,order:&Tensor<i64>,weights:&Tensor<S>,hist:&mut Tensor<S>,#[comptime] weighted:bool) {
    let bin=ABSOLUTE_POS as usize;
    if bin<hist.len() {
        let begin=key_bound(keys,i64::cast_from(bin),false);let end=key_bound(keys,i64::cast_from(bin),true);
        let mut value=S::new(0.0);
        for position in begin..end {
            let mut item=S::new(1.0);
            if comptime!(weighted) {
                let sample=usize::cast_from(order[address(order,position)]);item=weights[address(weights,sample)];
            }
            value=S::cast_from(value+item);
        }
        hist[address(hist,bin)]=value;
    }
}

#[ruda(launch)]
fn density<S:Float+RudaElement,F:Float+RudaElement>(hist:&mut Tensor<S>,total:&Tensor<S>,edges:&Tensor<S>,#[comptime] bins:Vec<u32>,#[comptime] dimensions:usize) {
    let index=ABSOLUTE_POS as usize;
    if index<hist.len() {
        let counts=Array::<u32>::from_data(bins);
        let mut value=S::cast_from(F::cast_from(hist[address(hist,index)])/F::cast_from(total[0]));
        let mut stride=hist.len();let mut edge_base=0usize;
        for axis in 0..dimensions {
            let count=counts[axis] as usize;stride/=count;let bin=index/stride%count;
            let width=S::cast_from(edges[address(edges,edge_base+bin+1)]-edges[address(edges,edge_base+bin)]);
            value=S::cast_from(F::cast_from(value)/F::cast_from(width));edge_base+=count+1;
        }
        hist[address(hist,index)]=value;
    }
}

fn grid(n:usize)->RudaCount {RudaCount::Static(u32::try_from(n.div_ceil(256)).expect("histogram launch overflow"),1,1)}

fn bin_shape(ints:&[i64])->(Vec<u32>,usize,usize) {
    let bins:Vec<u32>=ints.iter().map(|&n|u32::try_from(n).ok().filter(|&n|n>0).expect("histogram bins must be positive U32")).collect();
    let count=bins.iter().try_fold(1usize,|total,&n|total.checked_mul(n as usize)).filter(|&n|n<=u32::MAX as usize).expect("histogram output overflow");
    let edges=bins.iter().try_fold(0usize,|total,&n|total.checked_add(n as usize+1)).filter(|&n|n<=u32::MAX as usize).expect("histogram edge count overflow");
    (bins,count,edges)
}

pub(super) fn dispatch(op:u32,xs:&[View],ints:&[i64],values:&[f64]) {
    assert!((350..=353).contains(&op));let c=client();let block=RudaDim::new_1d(256);
    if op==350 {
        assert_eq!(xs.len(),2);assert_eq!(xs[1].shape.len(),1);assert!(xs[1].len>0);
        let floating=matches!(xs[0].dtype,0|9);
        if floating {assert!(ints.is_empty());assert_eq!(values.len(),2);assert_eq!(xs[0].dtype,xs[1].dtype);}
        else {assert!((4..=8).contains(&xs[0].dtype));assert_eq!(ints.len(),2);assert!(values.is_empty());assert_eq!(xs[1].dtype,4);}
        if xs[0].len==0 {return;}
        macro_rules! float {($s:ty)=>{unsafe {histc::launch::<$s,$s,$s,CudaRuntime>(&c,grid(xs[0].len),block,xs[0].arg(),xs[1].arg(),values[0] as $s,values[1] as $s);}};}
        macro_rules! integer {($i:ty)=>{unsafe {histc::launch::<$i,i64,i64,CudaRuntime>(&c,grid(xs[0].len),block,xs[0].arg(),xs[1].arg(),ints[0],ints[1]);}};}
        match xs[0].dtype {0=>float!(f32),9=>float!(f64),4=>integer!(i64),5=>integer!(i32),6=>integer!(i16),7=>integer!(i8),8=>integer!(u8),_=>unreachable!()}
        LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);return;
    }
    assert!(values.is_empty());let scalar;let mut bins=Vec::new();let mut dimensions=0usize;let mut linear=false;let mut weighted=false;let mut ordered=false;
    if op==351 {
        assert_eq!(xs.len(),3);assert!(!ints.is_empty());assert!(matches!(ints[0],0|1));linear=ints[0]!=0;
        assert_eq!(xs[0].shape.len(),2);assert_eq!(xs[1].shape.len(),1);assert_eq!(xs[2].shape.len(),1);
        dimensions=xs[0].shape[1];assert_eq!(ints.len(),dimensions+1);let (shape,_count,edges)=bin_shape(&ints[1..]);bins=shape;
        assert_eq!(xs[1].len,edges);assert_eq!(xs[2].len,xs[0].shape[0]);assert_eq!(xs[2].dtype,4);scalar=xs[0].dtype;assert_eq!(xs[1].dtype,scalar);
        if dimensions==0 || xs[2].len==0 {return;}
    } else if op==352 {
        assert_eq!(xs.len(),4);assert_eq!(ints.len(),2);assert!(ints.iter().all(|&n|matches!(n,0|1)));weighted=ints[0]!=0;ordered=ints[1]!=0;
        assert_eq!(xs[0].dtype,4);assert_eq!(xs[1].dtype,4);assert_eq!(xs[0].shape.len(),1);assert_eq!(xs[3].shape.len(),1);scalar=xs[3].dtype;assert_eq!(xs[2].dtype,scalar);
        if ordered {assert_eq!(xs[1].len,xs[0].len);}if weighted {assert_eq!(xs[2].len,xs[0].len);}
        if xs[3].len==0 || (!ordered && xs[0].len==0) {return;}
    } else {
        assert_eq!(xs.len(),3);let (shape,count,edges)=bin_shape(ints);dimensions=shape.len();bins=shape;
        if bins.is_empty() {bins.push(1);}
        assert_eq!(xs[0].shape.len(),1);assert_eq!(xs[0].len,count);assert_eq!(xs[1].len,1);assert_eq!(xs[2].len,edges);
        scalar=xs[0].dtype;assert_eq!(xs[1].dtype,scalar);assert_eq!(xs[2].dtype,scalar);
    }
    assert!(matches!(scalar,0|1|2|9));
    macro_rules! run {($s:ty,$f:ty)=>{unsafe {
        if op==351 {classify::launch::<$s,CudaRuntime>(&c,grid(xs[2].len),block,xs[0].arg(),xs[1].arg(),xs[2].arg(),bins.clone(),linear);}
        else if op==352 {
            if ordered {accumulate_ordered::launch::<$s,CudaRuntime>(&c,grid(xs[3].len),block,xs[0].arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),weighted);}
            else {accumulate::launch::<$s,CudaRuntime>(&c,grid(xs[0].len),block,xs[0].arg(),xs[2].arg(),xs[3].arg(),weighted);}
        } else {density::launch::<$s,$f,CudaRuntime>(&c,grid(xs[0].len),block,xs[0].arg(),xs[1].arg(),xs[2].arg(),bins.clone(),dimensions);}
    }};}
    match scalar {0=>run!(f32,f32),1=>run!(f16,f32),2=>run!(bf16,f32),9=>run!(f64,f64),_=>unreachable!()}
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);
}
