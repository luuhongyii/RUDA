// ATen special-polynomial recurrences; retained PyTorch license: bessel.rs.
use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn endpoint<F:Float>(n:i64,positive:bool,#[comptime] family:u32)->F {
    let one=F::new(1.0);let mut sign=one;if n%2i64!=0i64 {sign=-one;}
    let mut value=one;
    if comptime!(family==0) {if !positive {value=sign;}}
    else if comptime!(family==1) {value=F::cast_from(n+1i64);if !positive {value*=sign;}}
    else if comptime!(family==2) {if !positive {value=sign*F::cast_from(n+n+1i64);}}
    else {value=sign;if positive {value=F::cast_from(n+n+1i64);}}
    value
}

#[ruda]
fn chebyshev<F:Float>(x:F,n:i64,#[comptime] family:u32,#[comptime] shifted:bool)->F {
    let zero=F::new(0.0);let one=F::new(1.0);let two=F::new(2.0);let mut result=zero;
    if n>=0i64 {
        let mut z=x;let mut at_endpoint=x.abs()==one;let mut positive=x>zero;
        if comptime!(shifted) {z=x+x-one;at_endpoint=x==zero || x==one;positive=x==one;}
        let mut threshold=8i64;
        if comptime!(family==0 || shifted) {threshold=6i64;}
        if comptime!(family==3 && shifted) {threshold=4i64;}
        if at_endpoint {result=endpoint::<F>(n,positive,family);}
        else if n>threshold && z.abs()<one {
            let angle=z.acos();let degree=F::cast_from(n);
            if comptime!(family==0) {result=(degree*angle).cos();}
            else if comptime!(family==1) {
                let sine=angle.sin();let order=F::cast_from(n+1i64);
                if sine!=zero {result=(order*angle).sin()/sine;}
                else {result=order*(order*angle).cos()/z;}
            } else if comptime!(family==2) {
                if (angle/two).sin()!=one {result=((degree+F::new(0.5))*angle).cos()/(angle/two).cos();}
                else {result=endpoint::<F>(n,false,2u32);}
            } else {
                if (angle/two).cos()!=one {result=((degree+F::new(0.5))*angle).sin()/(angle/two).sin();}
                else {
                    let mut edge_positive=x>zero;if comptime!(shifted) {edge_positive=false;}
                    result=endpoint::<F>(n,edge_positive,3u32);
                }
            }
        } else if n==0i64 {result=one;}
        else {
            let mut previous=one;let mut current=z;
            if comptime!(family==1) {current=z+z;}
            else if comptime!(family==2) {current=z+z-one;}
            else if comptime!(family==3) {current=z+z+one;}
            let mut k=1i64;
            while k<n && current==current {
                let next=(z+z)*current-previous;previous=current;current=next;k+=1i64;
            }
            result=current;
        }
    }
    result
}

#[ruda]
fn hermite<F:Float>(x:F,n:i64,#[comptime] physicists:bool,#[comptime] limit:i64)->F {
    let one=F::new(1.0);let mut result=F::new(0.0);let mut factor=x;
    if comptime!(physicists) {factor=x+x;}
    if n<0i64 {}
    else if n==0i64 {result=one;}
    else if n==1i64 {result=factor;}
    else if n>limit {result=F::new(f32::NAN);}
    else {
        let mut previous=one;let mut current=factor;let mut k=1i64;
        while k<n {
            let mut coefficient=k;if comptime!(physicists) {coefficient=k+k;}
            let next=factor*current-F::cast_from(coefficient)*previous;
            previous=current;current=next;k+=1i64;
        }
        result=current;
    }
    result
}

#[ruda]
fn classical<F:Float>(x:F,n:i64,#[comptime] legendre:bool)->F {
    let one=F::new(1.0);let mut result=F::new(0.0);
    if n>=0i64 {
        let mut at_endpoint=x.abs()==F::new(0.0);let mut endpoint_value=one;
        if comptime!(legendre) {at_endpoint=x.abs()==one;endpoint_value=endpoint::<F>(n,x>F::new(0.0),0u32);}
        if at_endpoint || n==0i64 {result=endpoint_value;}
        else {
            let mut previous=one;let mut current=one-x;if comptime!(legendre) {current=x;}
            let mut k=1i64;
            while k<n && current==current {
                let mut next=F::new(0.0);
                if comptime!(legendre) {next=(F::cast_from(k+k+1i64)*x*current-F::cast_from(k)*previous)/F::cast_from(k+1i64);}
                else {next=((F::cast_from(k+k)+(one-x))*current-F::cast_from(k)*previous)/F::cast_from(k+1i64);}
                previous=current;current=next;k+=1i64;
            }
            result=current;
        }
    }
    result
}

#[ruda(launch)]
fn evaluate<F:Float+RudaElement>(x:&Tensor<F>,orders:&Tensor<F>,out:&mut Tensor<F>,#[comptime] kind:u32,#[comptime] limit:i64) {
    let index=ABSOLUTE_POS as usize;
    if index<out.len() {
        let value=x[address(x,index)];let order=orders[address(orders,index)];let mut n=i64::cast_from(order);
        if order!=order {n=i64::MIN;}
        let mut result=F::new(0.0);
        if comptime!(kind<8) {result=chebyshev::<F>(value,n,comptime!(kind%4),comptime!(kind>=4));}
        else if comptime!(kind<10) {result=hermite::<F>(value,n,comptime!(kind==8),limit);}
        else {result=classical::<F>(value,n,comptime!(kind==11));}
        out[address(out,index)]=result;
    }
}

pub(super) fn dispatch(op:u32,xs:&[View]) {
    assert!((368..=379).contains(&op));assert_eq!(xs.len(),3);let input=&xs[0];
    assert!(matches!(input.dtype,0|9));
    for tensor in &xs[1..] {assert_eq!(tensor.dtype,input.dtype);assert_eq!(tensor.shape,input.shape);}
    if input.len==0 {return;}
    let c=client();let grid=RudaCount::Static(u32::try_from(input.len.div_ceil(256)).expect("polynomial launch overflow"),1,1);let block=RudaDim::new_1d(256);
    macro_rules! run {($f:ty,$limit:expr)=>{unsafe {evaluate::launch::<$f,CudaRuntime>(&c,grid,block,xs[0].arg(),xs[1].arg(),xs[2].arg(),op-368,$limit);}};}
    if input.dtype==9 {run!(f64,512);}else{run!(f32,128);}
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);
}
