use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn map_index(position:i64, size:i64, padding:i64, #[comptime] reflect:bool)->i64 {
    let x=position-padding;
    let mut mapped=i64::min(i64::max(x,0),size-1);
    if comptime!(reflect) {
        let mut left=x;let mut right=x-size+1;
        if left<0 {left=-left;}
        if right<0 {right=-right;}
        mapped=left-right-x+size-1;
    }
    mapped
}

#[ruda(launch)]
fn forward<E:Numeric+RudaElement>(a:&Tensor<E>,out:&mut Tensor<E>,pw:i64,ph:i64,pd:i64,#[comptime] reflect:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let components=out.shape(5);let lane=p%components;let point=p/components;
        let ow=out.shape(4);let oh=out.shape(3);let od=out.shape(2);
        let iw=i64::cast_from(a.shape(4));let ih=i64::cast_from(a.shape(3));let id=i64::cast_from(a.shape(2));
        let x=map_index(i64::cast_from(point%ow),iw,pw,reflect);
        let y=map_index(i64::cast_from(point/ow%oh),ih,ph,reflect);
        let z=map_index(i64::cast_from(point/(ow*oh)%od),id,pd,reflect);
        let plane=i64::cast_from(point/(ow*oh*od));
        let source=usize::cast_from(((plane*id+z)*ih+y)*iw+x)*components+lane;
        out[address(out,p)]=a[address(a,source)];
    }
}

#[ruda(launch)]
fn clear<S:Float+RudaElement>(out:&mut Tensor<S>) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {out[address(out,p)]=S::new(0.0);}
}

#[ruda(launch)]
fn backward_scatter<S:Float+RudaElement>(grad:&Tensor<S>,out:&mut Tensor<Atomic<S>>,pw:i64,ph:i64,pd:i64,#[comptime] reflect:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<grad.len() {
        let components=grad.shape(5);let lane=p%components;let point=p/components;
        let ow=grad.shape(4);let oh=grad.shape(3);let od=grad.shape(2);let plane=point/(ow*oh*od);
        let x=usize::cast_from(map_index(i64::cast_from(point%ow),i64::cast_from(out.shape(4)),pw,reflect));
        let y=usize::cast_from(map_index(i64::cast_from(point/ow%oh),i64::cast_from(out.shape(3)),ph,reflect));
        let z=usize::cast_from(map_index(i64::cast_from(point/(ow*oh)%od),i64::cast_from(out.shape(2)),pd,reflect));
        let channels=out.shape(1);
        let offset=plane/channels*out.stride(0)+plane%channels*out.stride(1)
            +z*out.stride(2)+y*out.stride(3)+x*out.stride(4)+lane*out.stride(5);
        out[offset].fetch_add(grad[address(grad,p)]);
    }
}

#[ruda]
fn add_at<S:Float+RudaElement>(grad:&Tensor<S>,sum:S,plane:i64,y:i64,x:i64,lane:usize,#[comptime] bounds:u32)->S {
    let h=i64::cast_from(grad.shape(3));let w=i64::cast_from(grad.shape(4));let mut result=sum;
    let spatial=y*w+x;
    let mut inside=spatial<h*w;
    if comptime!(bounds==0) {inside=true;}
    else if comptime!(bounds==2) {inside=inside && spatial>=0;}
    if inside {
        let at=(plane*h*w+spatial)*i64::cast_from(grad.shape(5))+i64::cast_from(lane);
        if at<0 || at>=i64::cast_from(grad.len()) {
            intrinsic!(|scope| {scope.register(ruda_core::ir::Branch::Unreachable);});
        }
        result+=grad[address(grad,usize::cast_from(at))];
    }
    result
}

#[ruda(launch)]
fn reflection_backward_deterministic<S:Float+RudaElement>(grad:&Tensor<S>,out:&mut Tensor<S>,pw:i64,ph:i64) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let components=out.shape(5);let lane=p%components;let point=p/components;
        let iw=i64::cast_from(out.shape(4));let ih=i64::cast_from(out.shape(3));
        let x=i64::cast_from(point%out.shape(4));let y=i64::cast_from(point/out.shape(4)%out.shape(3));
        let plane=i64::cast_from(point/(out.shape(3)*out.shape(4)));
        let center_x=pw+x;let center_y=ph+y;let left=pw-x;let right=pw+2*(iw-1)-x;
        let top=ph-y;let bottom=ph+2*(ih-1)-y;let mut sum=S::new(0.0);
        let pr=i64::cast_from(grad.shape(4))-iw-pw;let pb=i64::cast_from(grad.shape(3))-ih-ph;
        let is_left=x>=1 && x<=pw;let is_right=x<iw-1 && x>=iw-pr-1;
        if y>=1 && y<=ph {
            sum=add_at(grad,sum,plane,top,center_x,lane,1);
            if is_left {sum=add_at(grad,sum,plane,top,left,lane,2);}
            else if is_right {sum=add_at(grad,sum,plane,top,right,lane,2);}
        }
        if y<ih-1 && y>=ih-pb-1 {
            sum=add_at(grad,sum,plane,bottom,center_x,lane,2);
            if is_left {sum=add_at(grad,sum,plane,bottom,left,lane,2);}
            else if is_right {sum=add_at(grad,sum,plane,bottom,right,lane,2);}
        }
        if is_left {sum=add_at(grad,sum,plane,center_y,left,lane,2);}
        if is_right {sum=add_at(grad,sum,plane,center_y,right,lane,2);}
        sum=add_at(grad,sum,plane,center_y,center_x,lane,0);
        out[address(out,p)]=S::new(0.0)+sum;
    }
}

fn component_view(v:&View)->View {
    let width=if v.dtype>=10 {2usize}else{1};
    let mut shape=v.shape.clone();shape.push(width);
    let mut strides:Vec<usize>=v.strides.iter().map(|s|s.checked_mul(width).expect("padding stride overflow")).collect();strides.push(1);
    let len=v.len.checked_mul(width).expect("padding size overflow");
    assert!(len<=u32::MAX as usize,"padding component count exceeds U32 indexing");
    if len!=0 {assert!(shape.iter().zip(&strides).try_fold(1usize,|span,(&n,&s)|
        (n-1).checked_mul(s).and_then(|term|span.checked_add(term))).is_some_and(|span|span<=u32::MAX as usize),
        "padding component span exceeds U32 indexing");}
    View {handle:v.handle.clone(),shape,strides,len,
        dtype:match v.dtype {10=>0,11=>9,code=>code}}
}

fn grid(n:usize)->RudaCount {
    assert!(n<=u32::MAX as usize,"padding component count exceeds U32 indexing");
    RudaCount::Static(u32::try_from(n.div_ceil(256)).expect("padding grid overflow"),1,1)
}

fn launch_forward<E:Numeric+RudaElement>(c:&ComputeClient<CudaRuntime>,a:&View,out:&View,p:&[i64]) {
    if out.len==0 {return;}
    unsafe {forward::launch::<E,CudaRuntime>(c,grid(out.len),RudaDim::new_1d(256),a.arg(),out.arg(),p[0],p[2],p[4],p[7]!=0);}
}

fn launch_backward<S:Float+RudaElement>(c:&ComputeClient<CudaRuntime>,grad:&View,out:&View,p:&[i64]) {
    if out.len==0 {return;}
    let block=RudaDim::new_1d(256);
    unsafe {
        if p[8]!=0 {
            reflection_backward_deterministic::launch::<S,CudaRuntime>(c,grid(out.len),block,grad.arg(),out.arg(),p[0],p[2]);
        } else {
            clear::launch::<S,CudaRuntime>(c,grid(out.len),block,out.arg());
            if grad.len!=0 {backward_scatter::launch::<S,CudaRuntime>(c,grid(grad.len),block,grad.arg(),out.arg(),p[0],p[2],p[4],p[7]!=0);}
        }
    }
}

pub(super) fn dispatch(op:u32,xs:&[View],p:&[i64]) {
    assert!(op==323 || op==324);assert_eq!(xs.len(),2);assert_eq!(p.len(),9);
    assert!((1..=3).contains(&p[6]));assert!(p[7]==0 || p[7]==1);assert!(p[8]==0 || p[8]==1);
    assert!(p[8]==0 || op==324 && p[6]==2 && p[7]==1);
    let a=&xs[0];let out=&xs[1];assert_eq!(a.dtype,out.dtype);
    assert!(matches!(a.dtype,0|1|2|4|5|6|7|8|9|10|11));
    assert_eq!(a.shape.len(),5);assert_eq!(out.shape.len(),5);assert_eq!(a.shape[..2],out.shape[..2]);
    let (input,padded)=if op==323 {(a,out)}else{(out,a)};
    assert!(input.shape[1..].iter().all(|&n|n>0));
    for axis in 0..3 {
        let n=i64::try_from(input.shape[4-axis]).expect("padding input dimension overflow");
        let size=n.checked_add(p[2*axis]).and_then(|v|v.checked_add(p[2*axis+1])).expect("padding size overflow");
        assert!(size>=0);assert_eq!(padded.shape[4-axis],size as usize);
        if p[7]!=0 {assert!(p[2*axis]<n && p[2*axis+1]<n);}
        if axis>=p[6] as usize {assert_eq!(n,1);assert_eq!(p[2*axis],0);assert_eq!(p[2*axis+1],0);}
    }
    let a=component_view(a);let out=component_view(out);let c=client();
    if op==323 {
        macro_rules! run {($e:ty)=>{launch_forward::<$e>(&c,&a,&out,p)};}
        match a.dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),4=>run!(i64),5=>run!(i32),6=>run!(i16),7=>run!(i8),8=>run!(u8),9=>run!(f64),_=>unreachable!()}
    } else {
        match a.dtype {0=>launch_backward::<f32>(&c,&a,&out,p),1=>launch_backward::<f16>(&c,&a,&out,p),2=>launch_backward::<bf16>(&c,&a,&out,p),9=>launch_backward::<f64>(&c,&a,&out,p),_=>panic!("padding backward requires floating or complex storage")}
    }
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);
}
