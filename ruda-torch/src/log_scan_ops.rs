// ATen logcumsumexp combine and scan rules; retained PyTorch license: bessel.rs.
use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn infinite_exp<F:Float>(angle:F)->(F,F) {
    let cosine=angle.cos();let sine=angle.sin();let inf=F::new(f32::INFINITY);
    let mut real=inf*cosine;let mut imag=inf*sine;
    if cosine==F::new(0.0) {real=F::new(0.0);}if sine==F::new(0.0) {imag=F::new(0.0);}
    (real,imag)
}

#[ruda]
fn add<S:Float,F:Float>(ar:S,ai:S,br:S,bi:S,#[comptime] complex:bool)->(S,S) {
    let x=F::cast_from(ar);let y=F::cast_from(br);let inf=F::new(f32::INFINITY);
    let mut real=x;let mut imag=F::new(0.0);
    if comptime!(complex) {
        let u=F::cast_from(ai);let v=F::cast_from(bi);
        real=F::new(f32::NAN);imag=real;
        if x==x && y==y && u==u && v==v {
            let mut low=y;let mut low_i=v;let mut high=x;let mut high_i=u;
            if x<y {low=x;low_i=u;high=y;high_i=v;}
            if low.abs()==inf && low==high {
                if low<F::new(0.0) {real=low;imag=low_i;}
                else {
                    let (a,b)=infinite_exp::<F>(low_i);let (c,d)=infinite_exp::<F>(high_i);
                    let (r,i)=crate::elementary::complex_log1p::<F>(a+c-F::new(1.0),b+d);real=r;imag=i;
                }
            } else {
                let magnitude=(low-high).exp();let angle=low_i-high_i;
                let (r,i)=crate::elementary::complex_log1p::<F>(magnitude*angle.cos(),magnitude*angle.sin());
                real=r+high;imag=i+high_i;
            }
        }
    } else {
        let mut low=x;let mut high=x;
        if y!=y {low=y;high=y;}
        else if x==x {if y<x {low=y;}if y>x {high=y;}}
        if low!=high || low.abs()!=inf {real=(low-high).exp().log1p()+high;}
    }
    (S::cast_from(real),S::cast_from(imag))
}

#[ruda(launch)]
fn outer<S:Float+RudaElement,F:Float+RudaElement>(ar:&Tensor<S>,ai:&Tensor<S>,out_r:&mut Tensor<S>,out_i:&mut Tensor<S>,#[comptime] complex:bool) {
    let row=ABSOLUTE_POS as usize;let inner=ar.shape(2);let width=ar.shape(1);
    if row<ar.shape(0)*inner {
        let base=row/inner*width*inner+row%inner;let mut real=S::new(f32::NEG_INFINITY);let mut imag=S::new(-0.0);
        for position in 0..width {
            let index=base+position*inner;let mut value_i=S::new(0.0);
            if comptime!(complex) {value_i=ai[address(ai,index)];}
            let (r,i)=add::<S,F>(real,imag,ar[address(ar,index)],value_i,complex);real=r;imag=i;
            out_r[address(out_r,index)]=real;if comptime!(complex) {out_i[address(out_i,index)]=imag;}
        }
    }
}

#[ruda(launch)]
fn inner<S:Float+RudaElement,F:Float+RudaElement>(ar:&Tensor<S>,ai:&Tensor<S>,out_r:&mut Tensor<S>,out_i:&mut Tensor<S>,#[comptime] complex:bool,#[comptime] threads:usize,#[comptime] single:bool) {
    let row=(RUDA_POS_X+RUDA_POS_Y*RUDA_COUNT_X) as usize;let lane=UNIT_POS as usize;
    if row<ar.shape(0) {
        let width=i64::cast_from(ar.shape(1));let base=row*ar.shape(1);
        let mut re=SharedMemory::<S>::new(threads*2);let mut im=SharedMemory::<S>::new(threads*2);
        let mut carry_r=S::new(f32::NEG_INFINITY);let mut carry_i=S::new(-0.0);let mut tile=0i64;
        while tile<width {
            let first=tile+i64::cast_from(lane);let second=first+i64::cast_from(threads);
            re[lane]=S::new(f32::NEG_INFINITY);re[lane+threads]=S::new(f32::NEG_INFINITY);
            im[lane]=S::new(-0.0);im[lane+threads]=S::new(-0.0);
            if first<width {
                re[lane]=ar[address(ar,base+usize::cast_from(first))];
                if comptime!(complex) {im[lane]=ai[address(ai,base+usize::cast_from(first))];}
            }
            if second<width {
                re[lane+threads]=ar[address(ar,base+usize::cast_from(second))];
                if comptime!(complex) {im[lane+threads]=ai[address(ai,base+usize::cast_from(second))];}
            }
            if lane==0 {
                let mut include=true;if comptime!(single) {include=tile>0;}
                if include {let (r,i)=add::<S,F>(re[0],im[0],carry_r,carry_i,complex);re[0]=r;im[0]=i;}
            }
            sync_ruda();let mut step=1usize;
            while step<=threads {
                let start=lane/step*(step*2)+step;let to=start+lane%step;let from=start-1;
                let (r,i)=add::<S,F>(re[to],im[to],re[from],im[from],complex);re[to]=r;im[to]=i;
                sync_ruda();step*=2;
            }
            if first<width {out_r[address(out_r,base+usize::cast_from(first))]=re[lane];if comptime!(complex) {out_i[address(out_i,base+usize::cast_from(first))]=im[lane];}}
            if second<width {out_r[address(out_r,base+usize::cast_from(second))]=re[lane+threads];if comptime!(complex) {out_i[address(out_i,base+usize::cast_from(second))]=im[lane+threads];}}
            carry_r=re[threads*2-1];carry_i=im[threads*2-1];sync_ruda();tile+=i64::cast_from(threads*2);
        }
    }
}

pub(super) fn dispatch(xs:&[View],ints:&[i64]) {
    assert_eq!(xs.len(),4);assert_eq!(ints.len(),2);assert!(ints.iter().all(|&n|matches!(n,0|1)));
    let input=&xs[0];assert_eq!(input.shape.len(),3);assert!(matches!(input.dtype,0|1|2|9));
    for value in &xs[1..] {assert_eq!(value.shape,input.shape);assert_eq!(value.dtype,input.dtype);}
    let complex=ints[0]!=0;let parallel=ints[1]!=0;
    if complex {assert!(matches!(input.dtype,0|9));}if input.len==0 {return;}
    let width=input.shape[1];let rows=input.shape[0]*input.shape[2];let single=input.len==width;
    let c=client();let mut threads=256usize;let grid;
    if parallel {
        assert_eq!(input.shape[2],1);
        let log_x=usize::BITS-(width-1).leading_zeros();let log_y=usize::BITS-(rows-1).leading_zeros();
        let log_threads=(9u32.wrapping_add(log_x.wrapping_sub(log_y))/2).clamp(4,9);threads=1usize<<log_threads;
        let count=u32::try_from(rows).expect("log scan row overflow");let x=count.min(1<<20);grid=RudaCount::Static(x,count.div_ceil(x),1);
    } else {grid=RudaCount::Static(u32::try_from(rows.div_ceil(threads)).expect("log scan launch overflow"),1,1);}
    let block=RudaDim::new_1d(threads as u32);
    macro_rules! run {($s:ty,$f:ty)=>{unsafe {
        if parallel {inner::launch::<$s,$f,CudaRuntime>(&c,grid,block,xs[0].arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),complex,threads,single);}
        else {outer::launch::<$s,$f,CudaRuntime>(&c,grid,block,xs[0].arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),complex);}
    }};}
    match input.dtype {0=>run!(f32,f32),1=>run!(f16,f32),2=>run!(bf16,f32),9=>run!(f64,f64),_=>unreachable!()}
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);
}
