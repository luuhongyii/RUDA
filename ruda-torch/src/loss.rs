// ATen CUDA loss formulas; retained PyTorch license: bessel.rs.
use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn stored<S:Float,F:Float>(x:F)->F {F::cast_from(S::cast_from(x))}

#[ruda(launch)]
fn pointwise<S:Float+RudaElement,F:Float+RudaElement>(a:&Tensor<S>,target:&Tensor<S>,grad:&Tensor<S>,out:&mut Tensor<S>,invalid:&mut Tensor<Atomic<u32>>,boundary:F,norm:F,#[comptime] op:u32) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let x=F::cast_from(a[address(a,p)]);let t=F::cast_from(target[address(target,p)]);
        let g=F::cast_from(grad[address(grad,p)]);let one=F::new(1.0);let zero=F::new(0.0);
        let d=stored::<S,F>(x-t);let b=stored::<S,F>(boundary);let n=stored::<S,F>(norm);
        let mut y=zero;
        if comptime!(op==293) {y=d*d;}
        else if comptime!(op==294) {y=stored::<S,F>(n*d)*g;}
        else if comptime!(op==295 || op==297) {
            let z=d.abs();let half_b=stored::<S,F>(F::new(0.5)*b);
            if z<b {y=stored::<S,F>(stored::<S,F>(F::new(0.5)*z)*z);if comptime!(op==295) {y=y/b;}}
            else {y=stored::<S,F>(z-half_b);if comptime!(op==297) {y=b*y;}}
        } else if comptime!(op==296 || op==298) {
            if d< -b {y=stored::<S,F>(-n*g);if comptime!(op==298) {y=y*b;}}
            else if d>b {y=stored::<S,F>(n*g);if comptime!(op==298) {y=y*b;}}
            else {y=stored::<S,F>(stored::<S,F>(n*d)*g);if comptime!(op==296) {y=y/b;}}
        } else if comptime!(op==299) {
            if !(x>=zero && x<=one && t>=zero && t<=one) {invalid[0].store(1u32);}
            let mut lx=stored::<S,F>(x.ln());let mut ln=stored::<S,F>((-x).log1p());
            if lx<F::new(-100.0) {lx=F::new(-100.0);}if ln<F::new(-100.0) {ln=F::new(-100.0);}
            y=stored::<S,F>(stored::<S,F>(t-one)*ln)-stored::<S,F>(t*lx);
        } else if comptime!(op==300) {
            let mut den=stored::<S,F>(stored::<S,F>(one-x)*x);let eps=stored::<S,F>(F::cast_from(1e-12f32));
            if den<eps {den=eps;}y=stored::<S,F>(g*d)/den;
        } else if comptime!(op==301) {
            let z=stored::<S,F>(-x*t);y=stored::<S,F>(z.exp()).log1p();
        } else if comptime!(op==302) {
            let z=stored::<S,F>(stored::<S,F>(-t*x).exp());
            y=stored::<S,F>(stored::<S,F>(t*z)*(-norm));
            y=stored::<S,F>(y/stored::<S,F>(one+z))*g;
        }
        out[address(out,p)]=S::cast_from(y);
    }
}

#[ruda(launch)]
fn nll_unreduced<S:Float+RudaElement>(a:&Tensor<S>,target:&Tensor<i64>,weight:&Tensor<S>,out:&mut Tensor<S>,invalid:&mut Tensor<Atomic<u32>>,ignore:i64,#[comptime] weighted:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<target.len() {
        let t=target[address(target,p)];let mut y=S::new(0.0);
        if t!=ignore {
            if t<0i64 || t>=i64::cast_from(a.shape(1)) {invalid[0].store(1u32);}
            else {
                let spatial=a.shape(2);let offset=p/spatial*a.stride(0)+usize::cast_from(t)*a.stride(1)+(p%spatial)*a.stride(2);
                let mut w=S::new(1.0);if comptime!(weighted) {w=weight[address(weight,usize::cast_from(t))];}
                y= -w*a[offset];
            }
        }
        out[address(out,p)]=y;
    }
}

#[ruda(launch)]
fn nll_single<S:Float+RudaElement,F:Float+RudaElement>(a:&Tensor<S>,target:&Tensor<i64>,weight:&Tensor<S>,out:&mut Tensor<S>,total:&mut Tensor<S>,invalid:&mut Tensor<Atomic<u32>>,ignore:i64,#[comptime] weighted:bool,#[comptime] mean:bool) {
    let t=target[0];let mut y=S::new(0.0);let mut w=S::new(0.0);
    if t!=ignore {
        if t<0i64 || t>=i64::cast_from(a.shape(1)) {invalid[0].store(1u32);}
        else {
            w=S::new(1.0);if comptime!(weighted) {w=weight[address(weight,usize::cast_from(t))];}
            let x=a[usize::cast_from(t)*a.stride(1)];
            y= -w*x;
            if comptime!(mean) {y= -x;if w==S::new(0.0) {y=S::cast_from(F::new(f32::NAN));}}
        }
    }
    out[0]=y;total[0]=w;
}

#[ruda(launch)]
fn nll_reduce<S:Float+RudaElement,F:Float+RudaElement>(a:&Tensor<S>,target:&Tensor<i64>,weight:&Tensor<S>,partial:&mut Tensor<F>,invalid:&mut Tensor<Atomic<u32>>,ignore:i64,blocks:usize,#[comptime] threads:usize,#[comptime] spatial:bool,#[comptime] weighted:bool) {
    let lane=UNIT_POS as usize;let block=RUDA_POS_X as usize;
    let mut shared_loss=SharedMemory::<F>::new(threads);let mut shared_weight=SharedMemory::<F>::new(threads);
    let mut start=lane;let mut end=target.len();let mut step=usize::cast_from(threads);
    if comptime!(spatial) {let sample=block/blocks;start=sample*a.shape(2)+(block%blocks)*threads+lane;end=(sample+1)*a.shape(2);step=threads*blocks;}
    let mut loss=F::new(0.0);let mut total=F::new(0.0);let mut p=start;
    while p<end {
        let t=target[address(target,p)];
        if t!=ignore {
            if t<0i64 || t>=i64::cast_from(a.shape(1)) {invalid[0].store(1u32);}
            else {
                let width=a.shape(2);let offset=p/width*a.stride(0)+usize::cast_from(t)*a.stride(1)+(p%width)*a.stride(2);
                let mut w=S::new(1.0);if comptime!(weighted) {w=weight[address(weight,usize::cast_from(t))];}
                loss-=F::cast_from(a[offset]*w);total+=F::cast_from(w);
            }
        }
        p+=step;
    }
    shared_loss[lane]=loss;shared_weight[lane]=total;sync_ruda();
    let mut stride=usize::cast_from(threads/2);
    while stride>0 {if lane<stride {shared_loss[lane]=shared_loss[lane]+shared_loss[lane+stride];shared_weight[lane]=shared_weight[lane]+shared_weight[lane+stride];}sync_ruda();stride/=2;}
    if lane==0 {
        let mut value=shared_loss[0];let mut count=shared_weight[0];
        if comptime!(spatial) {value=stored::<S,F>(value);count=stored::<S,F>(count);}
        partial[block*2]=value;partial[block*2+1]=count;
    }
}

#[ruda(launch)]
fn nll_finish<S:Float+RudaElement,F:Float+RudaElement>(partial:&Tensor<F>,out:&mut Tensor<S>,total:&mut Tensor<S>,#[comptime] spatial:bool,#[comptime] mean:bool) {
    let mut value=F::new(0.0);let mut count=F::new(0.0);
    for i in 0..partial.len()/2 {
        value+=partial[i*2];count+=partial[i*2+1];
        if comptime!(spatial) {value=stored::<S,F>(value);count=stored::<S,F>(count);}
    }
    total[0]=S::cast_from(count);if comptime!(mean) {value=value/count;}out[0]=S::cast_from(value);
}

#[ruda(launch)]
fn nll_backward<S:Float+RudaElement>(target:&Tensor<i64>,weight:&Tensor<S>,grad:&Tensor<S>,total:&Tensor<S>,out:&mut Tensor<S>,invalid:&mut Tensor<Atomic<u32>>,ignore:i64,#[comptime] weighted:bool,#[comptime] reduction:u32) {
    let p=ABSOLUTE_POS as usize;
    if p<target.len() {
        let t=target[address(target,p)];
        if t!=ignore {
            if t<0i64 || t>=i64::cast_from(out.shape(1)) {invalid[0].store(1u32);}
            else {
                let mut g=grad[0];if comptime!(reduction==0) {g=grad[address(grad,p)];}
                if comptime!(reduction==1) {g=g/total[0];}
                g= -g;if comptime!(weighted) {g=weight[address(weight,usize::cast_from(t))]*g;}
                let width=out.shape(2);let offset=p/width*out.stride(0)+usize::cast_from(t)*out.stride(1)+(p%width)*out.stride(2);
                out[offset]=g;
            }
        }
    }
}

fn grid(len:usize)->RudaCount {RudaCount::Static(u32::try_from(len.div_ceil(128)).expect("loss grid overflow"),1,1)}
fn status(c:&ComputeClient<CudaRuntime>)->View {
    let value=View::packed(c.empty(4),vec![1],5);let bytes=value.byte_view();
    unsafe {kernels::fill_bytes::launch::<CudaRuntime>(c,RudaCount::Static(1,1,1),RudaDim::new_1d(4),bytes.arg(),0,0);}
    value
}
fn check(c:&ComputeClient<CudaRuntime>,value:&View,message:&str) {
    let bytes=c.read_one(value.handle.clone()).expect("loss validation readback failed");DOWNLOAD.fetch_add(4,Ordering::Relaxed);
    assert_eq!(u32::from_le_bytes(bytes[..4].try_into().unwrap()),0,"{message}");
}

fn nll_dispatch<S:Float+RudaElement,F:Float+RudaElement>(c:&ComputeClient<CudaRuntime>,op:u32,xs:&[View],ints:&[i64]) {
    let a=&xs[0];let target=&xs[1];let weight=&xs[2];let out=xs.last().unwrap();
    let reduction=ints[0] as u32;let ignore=ints[1];let mode=ints[2];let weighted=ints[3]!=0;
    let invalid=status(c);let dim=RudaDim::new_1d(128);
    if op==304 {
        if target.len>0 {unsafe {nll_backward::launch::<S,CudaRuntime>(c,grid(target.len),dim,target.arg(),weight.arg(),xs[3].arg(),xs[4].arg(),out.arg(),invalid.arg(),ignore,weighted,reduction);}}
    } else if mode==0 {
        unsafe {nll_single::launch::<S,F,CudaRuntime>(c,RudaCount::Static(1,1,1),RudaDim::new_1d(1),a.arg(),target.arg(),weight.arg(),xs[3].arg(),xs[4].arg(),invalid.arg(),ignore,weighted,reduction==1);}
    } else if reduction==0 {
        if target.len>0 {unsafe {nll_unreduced::launch::<S,CudaRuntime>(c,grid(target.len),dim,a.arg(),target.arg(),weight.arg(),xs[3].arg(),invalid.arg(),ignore,weighted);}}
    } else {
        let spatial=mode==2;let blocks=if spatial {a.shape[2].div_ceil(1024).div_euclid(128).max(1)}else{1};
        let count=if spatial {a.shape[0]*blocks}else{1};
        let threads=if spatial {1024}else{((target.len.max(16)/16) as f64).log2().round().exp2().clamp(32.0,1024.0) as usize};
        let partial=View::packed(c.empty((count*2*std::mem::size_of::<F>()).max(1)),vec![count*2],if a.dtype==9 {9}else{0});
        if count>0 {unsafe {nll_reduce::launch::<S,F,CudaRuntime>(c,RudaCount::Static(u32::try_from(count).expect("NLL grid overflow"),1,1),RudaDim::new_1d(threads as u32),a.arg(),target.arg(),weight.arg(),partial.arg(),invalid.arg(),ignore,blocks,threads,spatial,weighted);}}
        unsafe {nll_finish::launch::<S,F,CudaRuntime>(c,RudaCount::Static(1,1,1),RudaDim::new_1d(1),partial.arg(),xs[3].arg(),xs[4].arg(),spatial,reduction==1);}
    }
    check(c,&invalid,"NLL target is out of bounds");
}

pub(super) fn dispatch(op:u32,xs:&[View],ints:&[i64],values:&[f64]) {
    let c=client().fixed_execution_queue();let a=&xs[0];
    if op<=302 {
        assert_eq!(xs.len(),4);assert_eq!(values.len(),2);
        for x in xs {assert_eq!(a.shape,x.shape);assert_eq!(a.dtype,x.dtype);}
        if a.len==0 {return;}
        let invalid=if op==299 {status(&c)}else{View::packed(c.empty(4),vec![1],5)};
        macro_rules! run {($s:ty,$f:ty)=>{unsafe {pointwise::launch::<$s,$f,CudaRuntime>(&c,grid(a.len),RudaDim::new_1d(128),a.arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),invalid.arg(),values[0] as $f,values[1] as $f,op);}}}
        match a.dtype {0=>run!(f32,f32),1=>run!(f16,f32),2=>run!(bf16,f32),9=>run!(f64,f64),_=>panic!("loss requires real floating input")}
        if op==299 {check(&c,&invalid,"all BCE input and target elements must be between 0 and 1");}
    } else {
        assert_eq!(xs.len(),if op==303 {5}else{6});assert_eq!(ints.len(),4);
        assert!((0..=2).contains(&ints[0]) && (0..=2).contains(&ints[2]));
        assert_eq!(a.shape.len(),3);assert_eq!(xs[1].dtype,4);assert_eq!(xs[1].len,a.shape[0]*a.shape[2]);
        for x in &xs[2..] {assert_eq!(a.dtype,x.dtype);}
        if ints[3]!=0 {assert_eq!(xs[2].len,a.shape[1]);}
        if op==304 {assert_eq!(xs[5].shape,a.shape);assert_eq!(xs[4].len,1);assert_eq!(xs[3].len,if ints[0]==0 {xs[1].len}else{1});}
        else {assert_eq!(xs[4].len,1);assert_eq!(xs[3].len,if ints[0]==0 && ints[2]!=0 {xs[1].len}else{1});}
        match a.dtype {0=>nll_dispatch::<f32,f32>(&c,op,xs,ints),1=>nll_dispatch::<f16,f32>(&c,op,xs,ints),2=>nll_dispatch::<bf16,f32>(&c,op,xs,ints),9=>nll_dispatch::<f64,f64>(&c,op,xs,ints),_=>panic!("NLL requires real floating input")}
    }
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);
}
