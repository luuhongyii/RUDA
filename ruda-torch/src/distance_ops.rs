use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn triangle(index:usize,rows:usize)->(usize,usize) {
    let n=i64::cast_from(rows);let k=i64::cast_from(index);let half=f64::cast_from(rows)-0.5;
    let i=i64::cast_from(half-(half*half-1.0-2.0*f64::cast_from(index)).sqrt());
    let j=k-n*i+i*(i+1)/2+i+1;
    (usize::cast_from(i),usize::cast_from(j))
}

#[ruda]
fn combine<F:Float+RudaElement>(a:F,b:F,#[comptime] maximum:bool)->F {
    let mut value=a;
    if comptime!(maximum) {if b>a {value=b;}}
    else {value+=b;}
    value
}

#[ruda]
fn warp_reduce<F:Float+RudaElement>(value:F,#[comptime] maximum:bool)->F {
    let mut sum=value;let mut offset=16u32;
    while offset>0 {sum=combine(sum,plane_shuffle_down(sum,offset),maximum);offset/=2;}
    sum
}

#[ruda]
fn block_reduce<F:Float+RudaElement>(value:F,#[comptime] maximum:bool)->F {
    let lane=UNIT_POS as usize;let mut shared=SharedMemory::<F>::new(8usize);
    let mut sum=warp_reduce(value,maximum);sync_ruda();
    if lane%32==0 {shared[lane/32]=sum;}
    sync_ruda();sum=F::new(0.0);
    if lane<8 {sum=shared[lane];}
    if lane<32 {sum=warp_reduce(sum,maximum);}
    sum
}

#[ruda(launch)]
fn forward<F:Float+RudaElement>(a:&Tensor<F>,b:&Tensor<F>,out:&mut Tensor<F>,p:F,#[comptime] paired:bool,#[comptime] kind:u32) {
    let task=RUDA_POS_X as usize;let lane=UNIT_POS as usize;
    let rows=a.shape(a.rank()-2);let columns=b.shape(b.rank()-2);let features=a.shape(a.rank()-1);
    let mut batch=0usize;let mut i=0usize;let mut j=0usize;
    if comptime!(paired) {let (left,right)=triangle(task,rows);i=left;j=right;}
    else {batch=task/(rows*columns);i=task/columns%rows;j=task%columns;}
    let mut value=F::new(0.0);let mut feature=lane;
    while feature<features {
        let left=a[address(a,(batch*rows+i)*features+feature)];
        let right=b[address(b,(batch*columns+j)*features+feature)];let diff=(left-right).abs();
        if comptime!(kind==0) {
            if diff!=diff {value=diff;}else if diff!=F::new(0.0) {value+=F::new(1.0);}
        } else if comptime!(kind==1) {value+=diff;}
        else if comptime!(kind==2) {value+=diff*diff;}
        else if comptime!(kind==3) {if diff>value {value=diff;}}
        else {value+=diff.powf(p);}
        feature+=256;
    }
    value=block_reduce(value,kind==3);
    if lane==0 {
        if comptime!(kind==2) {value=value.sqrt();}
        else if comptime!(kind>=4) {value=value.powf(F::new(1.0)/p);}
        out[address(out,task)]=value;
    }
}

#[ruda]
fn contribution<F:Float+RudaElement>(diff:F,grad:F,distance:F,p:F,#[comptime] kind:u32)->F {
    let sign=F::cast_from(diff>F::new(0.0))-F::cast_from(diff<F::new(0.0));let mut value=F::new(0.0);
    if comptime!(kind==1) {value=grad*sign;}
    else if comptime!(kind==2) {if distance!=F::new(0.0) {value=grad*diff/distance;}}
    else if comptime!(kind==3) {value=grad*sign*F::cast_from(diff.abs()==distance);}
    else if comptime!(kind==4) {
        if distance!=F::new(0.0) && !(diff==F::new(0.0) && p<F::new(1.0)) {
            value=sign*diff.abs().powf(p-F::new(1.0))*grad/distance.powf(p-F::new(1.0));
        }
    } else if comptime!(kind==5) {
        if distance!=F::new(0.0) {value=diff*diff.abs().powf(p-F::new(2.0))*grad/distance.powf(p-F::new(1.0));}
    }
    value
}

#[ruda(launch)]
fn backward_terms<F:Float+RudaElement>(a:&Tensor<F>,b:&Tensor<F>,grad:&Tensor<F>,distance:&Tensor<F>,out:&mut Tensor<F>,p:F,#[comptime] paired:bool,#[comptime] kind:u32) {
    let index=ABSOLUTE_POS as usize;
    if index<out.len() {
        let features=a.shape(a.rank()-1);let rows=a.shape(a.rank()-2);let columns=b.shape(b.rank()-2);
        let feature=index%features;let mut batch=0usize;let mut i=0usize;let mut j=0usize;let mut pair=0usize;let mut negative=false;
        if comptime!(paired) {
            let slot=index/(features*rows);let row=index/features%rows;
            if slot<rows-row-1 {i=row;j=row+slot+1;}
            else {i=rows-2-slot;j=row;negative=true;}
            let first=i64::cast_from(i);let n=i64::cast_from(rows);
            pair=usize::cast_from(first*(2*n-first-1)/2+i64::cast_from(j-i-1));
        } else {
            i=index/features%rows;j=index/(features*rows)%columns;batch=index/(features*rows*columns);
            pair=(batch*rows+i)*columns+j;
        }
        let diff=a[address(a,(batch*rows+i)*features+feature)]-b[address(b,(batch*columns+j)*features+feature)];
        let mut value=contribution(diff,grad[address(grad,pair)],distance[address(distance,pair)],p,kind);
        if negative {value=-value;}
        out[address(out,index)]=value;
    }
}

fn pair_count(n:usize)->usize {n.checked_mul(n.saturating_sub(1)).expect("pair count overflow")/2}

pub(super) fn dispatch(op:u32,xs:&[View],ints:&[i64],scalars:&[f64]) {
    assert!(op==339 || op==340);let back=op==340;assert_eq!(xs.len(),if back {5}else{3});
    assert_eq!(ints.len(),1);assert!(ints[0]==0 || ints[0]==1);assert_eq!(scalars.len(),1);let paired=ints[0]!=0;
    let a=&xs[0];let b=&xs[1];let out=xs.last().unwrap();let rank=a.shape.len();assert!(rank>=2);
    assert!(a.dtype==0 || a.dtype==9);for view in xs {assert_eq!(view.dtype,a.dtype);}
    assert_eq!(rank,b.shape.len());assert_eq!(a.shape[..rank-2],b.shape[..rank-2]);assert_eq!(a.shape[rank-1],b.shape[rank-1]);
    let rows=a.shape[rank-2];let columns=b.shape[rank-2];let features=a.shape[rank-1];
    let batch=a.shape[..rank-2].iter().try_fold(1usize,|n,&d|n.checked_mul(d)).expect("distance batch overflow");
    let mut output_shape=a.shape[..rank-2].to_vec();output_shape.extend([rows,columns]);
    if paired {assert_eq!(rank,2);assert_eq!(a.shape,b.shape);output_shape=vec![pair_count(rows)];}
    if back {
        assert_eq!(xs[2].shape,output_shape);assert_eq!(xs[3].shape,output_shape);
        let expected=if paired {vec![rows.saturating_sub(1),rows,features]}else{vec![batch,columns,rows,features]};
        assert_eq!(out.shape,expected);
    } else {assert_eq!(out.shape,output_shape);}
    if out.len==0 {return;}
    assert!(features>0);let p=scalars[0];if !paired && !back {assert!(p>=0.0);}
    let kind=if p==0.0 {0}else if p==1.0 {1}else if back && p<2.0 {4}else if p==2.0 {2}else if p.is_infinite() {3}else{5};
    let c=client();let count=if back {out.len.div_ceil(256)}else{out.len};
    let launch=RudaCount::Static(u32::try_from(count).expect("distance launch overflow"),1,1);let block=RudaDim::new_1d(256);
    macro_rules! run {($f:ty)=>{unsafe {
        if back {backward_terms::launch::<$f,CudaRuntime>(&c,launch,block,a.arg(),b.arg(),xs[2].arg(),xs[3].arg(),out.arg(),p as $f,paired,kind);}
        else {forward::launch::<$f,CudaRuntime>(&c,launch,block,a.arg(),b.arg(),out.arg(),p as $f,paired,kind);}
    }};}
    match a.dtype {0=>run!(f32),9=>run!(f64),_=>unreachable!()}
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);
}
