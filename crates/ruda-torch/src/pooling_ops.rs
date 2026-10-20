use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn start(i:i64,output:i64,input:i64)->i64 {(i/output)*input+((i%output)*input)/output}

#[ruda]
fn end(i:i64,output:i64,input:i64)->i64 {1+((i+1)*input-1)/output}

#[ruda]
fn window(i:i64,input:i64,output:i64,k:i64,s:i64,p:i64,d:i64,#[comptime] adaptive:bool,#[comptime] maximum:bool)->(i64,i64,i64) {
    let mut first=0i64;let mut last=0i64;let mut count=0i64;
    if comptime!(adaptive) {first=start(i,output,input);last=end(i,output,input);count=last-first;}
    else {
        first=i*s-p;
        if comptime!(maximum) {
            last=i64::min(first+(k-1)*d+1,input);
            while first<0 {first+=d;}
            count=(last-first+d-1)/d;
        } else {
            last=i64::min(first+k,input+p);count=last-first;
            first=i64::max(first,0);last=i64::min(last,input);
        }
    }
    (first,last,count)
}

#[ruda]
fn inverse_window(i:i64,input:i64,output:i64,k:i64,s:i64,p:i64,d:i64,#[comptime] adaptive:bool)->(i64,i64) {
    let mut first=0i64;let mut last=0i64;
    if comptime!(adaptive) {first=start(i,input,output);last=end(i,input,output);}
    else {
        let at=i+p;let extent=(k-1)*d+1;
        if at>=extent {first=(at-extent)/s+1;}
        last=i64::min(at/s+1,output);
    }
    (first,last)
}

#[ruda(launch)]
fn forward<S:Float+RudaElement,F:Float+RudaElement>(
    a:&Tensor<S>,out:&mut Tensor<S>,indices:&mut Tensor<i64>,
    kd:i64,kh:i64,kw:i64,sd:i64,sh:i64,sw:i64,pd:i64,ph:i64,pw:i64,dd:i64,dh:i64,dw:i64,divisor:i64,
    #[comptime] include_pad:bool,#[comptime] adaptive:bool,#[comptime] maximum:bool,#[comptime] rank:u32,#[comptime] channels_last:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let id=i64::cast_from(a.shape(2));let ih=i64::cast_from(a.shape(3));let iw=i64::cast_from(a.shape(4));
        let od=i64::cast_from(out.shape(2));let oh=i64::cast_from(out.shape(3));let ow=i64::cast_from(out.shape(4));
        let pos=i64::cast_from(p);let w=pos%ow;let h=pos/ow%oh;let d=pos/(ow*oh)%od;let plane=pos/(od*oh*ow);
        let (d0,d1,dcount)=window(d,id,od,kd,sd,pd,dd,adaptive,maximum);
        let (h0,h1,hcount)=window(h,ih,oh,kh,sh,ph,dh,adaptive,maximum);
        let (w0,w1,wcount)=window(w,iw,ow,kw,sw,pw,dw,adaptive,maximum);
        let mut total=F::new(0.0);let mut best=F::new(f32::NEG_INFINITY);let mut index=(d0*ih+h0)*iw+w0;
        if comptime!(maximum && !adaptive && rank==2 && channels_last) {index=0;}
        let mut z=d0;
        while z<d1 {
            let mut y=h0;
            while y<h1 {
                let mut x=w0;
                while x<w1 {
                    let at=(z*ih+y)*iw+x;let value=F::cast_from(a[address(a,usize::cast_from(plane*id*ih*iw+at))]);
                    if comptime!(maximum) {if value>best || value!=value {best=value;index=at;}}
                    else {total+=value;}
                    x+=dw;
                }
                y+=dh;
            }
            z+=dd;
        }
        if comptime!(maximum) {out[address(out,p)]=S::cast_from(best);indices[address(indices,p)]=index;}
        else {
            let mut result=F::new(0.0);
            if d0<d1 && h0<h1 && w0<w1 {
                if comptime!(adaptive && rank==2) {
                    if comptime!(channels_last) {let factor=S::new(1.0)/S::cast_from((h1-h0)*(w1-w0));result=total*F::cast_from(factor);}
                    else {result=total/F::cast_from(h1-h0)/F::cast_from(w1-w0);}
                } else {
                    let mut divide=(d1-d0)*(h1-h0)*(w1-w0);
                    if comptime!(!adaptive && include_pad) {divide=dcount*hcount*wcount;}
                    if divisor!=0 {divide=divisor;}
                    result=total/F::cast_from(divide);
                }
            }
            out[address(out,p)]=S::cast_from(result);
        }
    }
}

#[ruda(launch)]
fn backward_gather<S:Float+RudaElement,F:Float+RudaElement>(
    grad:&Tensor<S>,indices:&Tensor<i64>,out:&mut Tensor<S>,
    kd:i64,kh:i64,kw:i64,sd:i64,sh:i64,sw:i64,pd:i64,ph:i64,pw:i64,dd:i64,dh:i64,dw:i64,divisor:i64,
    #[comptime] include_pad:bool,#[comptime] adaptive:bool,#[comptime] maximum:bool,#[comptime] rank:u32,#[comptime] channels_last:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let id=i64::cast_from(out.shape(2));let ih=i64::cast_from(out.shape(3));let iw=i64::cast_from(out.shape(4));
        let od=i64::cast_from(grad.shape(2));let oh=i64::cast_from(grad.shape(3));let ow=i64::cast_from(grad.shape(4));
        let pos=i64::cast_from(p);let w=pos%iw;let h=pos/iw%ih;let d=pos/(iw*ih)%id;let plane=pos/(id*ih*iw);
        let (d0,d1)=inverse_window(d,id,od,kd,sd,pd,dd,adaptive);
        let (h0,h1)=inverse_window(h,ih,oh,kh,sh,ph,dh,adaptive);
        let (w0,w1)=inverse_window(w,iw,ow,kw,sw,pw,dw,adaptive);
        let mut total=F::new(0.0);let mut rounded=S::new(0.0);
        for z in d0..d1 {for y in h0..h1 {for x in w0..w1 {
            let q=usize::cast_from(((plane*od+z)*oh+y)*ow+x);let g=grad[address(grad,q)];
            if comptime!(maximum) {
                if indices[address(indices,q)]==pos-plane*id*ih*iw {total+=F::cast_from(g);}
            } else {
                if comptime!(!adaptive && rank==3) {total+=F::cast_from(g);}
                else {
                    let (za,zb,zc)=window(z,id,od,kd,sd,pd,dd,adaptive,false);
                    let (ya,yb,yc)=window(y,ih,oh,kh,sh,ph,dh,adaptive,false);
                    let (xa,xb,xc)=window(x,iw,ow,kw,sw,pw,dw,adaptive,false);
                    if comptime!(adaptive) {
                        let mut contribution=S::cast_from(F::cast_from(g)/F::cast_from((zb-za)*(yb-ya)*(xb-xa)));
                        if comptime!(rank==2 && channels_last) {
                            let rh=S::new(1.0)/S::cast_from(yb-ya);let rw=S::new(1.0)/S::cast_from(xb-xa);
                            let factor=S::cast_from(F::cast_from(rw)*F::cast_from(rh));
                            contribution=S::cast_from(F::cast_from(g)*F::cast_from(factor));
                        }
                        rounded=S::cast_from(F::cast_from(rounded)+F::cast_from(contribution));
                    } else {
                        let mut divide=(zb-za)*(yb-ya)*(xb-xa);if comptime!(include_pad) {divide=zc*yc*xc;}
                        if divisor!=0 {divide=divisor;}
                        total+=F::cast_from(g)/F::cast_from(divide);
                    }
                }
            }
        }}}
        if comptime!(adaptive && !maximum) {out[address(out,p)]=rounded;}
        else {
            if comptime!(!maximum && rank==3) {let mut divide=kd*kh*kw;if divisor!=0 {divide=divisor;}total*=F::new(1.0)/F::cast_from(divide);}
            out[address(out,p)]=S::cast_from(total);
        }
    }
}

#[ruda(launch)]
fn clear<S:Numeric+RudaElement>(out:&mut Tensor<S>) {
    let p=ABSOLUTE_POS as usize;if p<out.len() {out[address(out,p)]=S::cast_from(0u32);}
}

#[ruda(launch)]
fn backward_scatter<S:Float+RudaElement,F:Float+RudaElement>(
    grad:&Tensor<S>,indices:&Tensor<i64>,out:&mut Tensor<Atomic<S>>,
    kd:i64,kh:i64,kw:i64,sd:i64,sh:i64,sw:i64,pd:i64,ph:i64,pw:i64,divisor:i64,
    #[comptime] include_pad:bool,#[comptime] adaptive:bool,#[comptime] maximum:bool,#[comptime] rank:u32) {
    let p=ABSOLUTE_POS as usize;
    if p<grad.len() {
        let id=i64::cast_from(out.shape(2));let ih=i64::cast_from(out.shape(3));let iw=i64::cast_from(out.shape(4));
        let od=i64::cast_from(grad.shape(2));let oh=i64::cast_from(grad.shape(3));let ow=i64::cast_from(grad.shape(4));
        let pos=i64::cast_from(p);let w=pos%ow;let h=pos/ow%oh;let d=pos/(ow*oh)%od;let plane=pos/(od*oh*ow);
        let channels=out.shape(1);let base=usize::cast_from(plane)/channels*out.stride(0)+usize::cast_from(plane)%channels*out.stride(1);
        let g=grad[address(grad,p)];
        if comptime!(maximum) {
            let index=indices[address(indices,p)];
            if index>=0 && index<id*ih*iw {
                let at=base+usize::cast_from(index/(ih*iw))*out.stride(2)+usize::cast_from(index/iw%ih)*out.stride(3)+usize::cast_from(index%iw)*out.stride(4);
                out[at].fetch_add(g);
            }
        } else {
            let (d0,d1,dc)=window(d,id,od,kd,sd,pd,1,adaptive,false);
            let (h0,h1,hc)=window(h,ih,oh,kh,sh,ph,1,adaptive,false);
            let (w0,w1,wc)=window(w,iw,ow,kw,sw,pw,1,adaptive,false);
            let mut contribution=S::new(0.0);
            if comptime!(adaptive) {
                if comptime!(rank==2) {contribution=g/S::cast_from(w1-w0)/S::cast_from(h1-h0);}
                else {contribution=g/S::cast_from(d1-d0)/S::cast_from(h1-h0)/S::cast_from(w1-w0);}
            } else {
                let mut divide=(d1-d0)*(h1-h0)*(w1-w0);if comptime!(include_pad) {divide=dc*hc*wc;}
                if divisor!=0 {divide=divisor;}
                contribution=S::cast_from(F::cast_from(g)/F::cast_from(divide));
            }
            for z in d0..d1 {for y in h0..h1 {for x in w0..w1 {
                let at=base+usize::cast_from(z)*out.stride(2)+usize::cast_from(y)*out.stride(3)+usize::cast_from(x)*out.stride(4);
                out[at].fetch_add(contribution);
            }}}
        }
    }
}

fn launch<S:Float+RudaElement,F:Float+RudaElement>(c:&ComputeClient<CudaRuntime>,op:u32,xs:&[View],p:&[i64]) {
    let out=&xs[if op==318 {1}else{3}];if out.len==0 {return;}
    let grid=|n:usize|RudaCount::Static(u32::try_from(n.div_ceil(256)).expect("pooling grid overflow"),1,1);
    let block=RudaDim::new_1d(256);let adaptive=p[14]!=0;let maximum=p[15]!=0;let rank=p[16] as u32;let channels_last=p[17]!=0;
    if op==318 {
        unsafe {forward::launch::<S,F,CudaRuntime>(c,grid(out.len),block,xs[0].arg(),out.arg(),xs[2].arg(),
            p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13]!=0,adaptive,maximum,rank,channels_last);}
    } else {
        let grad=&xs[1];let divisible=(2..5).all(|i|out.shape[i]%grad.shape[i]==0);
        let gather=if maximum {!adaptive && rank==2}
            else if adaptive {if rank==2 {channels_last}else{divisible}}
            else {rank==2 || (p[3..6]==[1,1,1] && p[6..9]==[0,0,0])};
        if gather {
            unsafe {backward_gather::launch::<S,F,CudaRuntime>(c,grid(out.len),block,grad.arg(),xs[2].arg(),out.arg(),
                p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13]!=0,adaptive,maximum,rank,channels_last);}
        } else {
            unsafe {
                clear::launch::<S,CudaRuntime>(c,grid(out.len),block,out.arg());
                if grad.len>0 {backward_scatter::launch::<S,F,CudaRuntime>(c,grid(grad.len),block,grad.arg(),xs[2].arg(),out.arg(),
                    p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[12],p[13]!=0,adaptive,maximum,rank);}
            }
        }
    }
}

pub(super) fn dispatch(op:u32,xs:&[View],p:&[i64]) {
    assert!(op==318 || op==319);assert_eq!(xs.len(),if op==318 {3}else{4});assert_eq!(p.len(),18);
    assert!(p[..6].iter().all(|&x|x>0));assert!(p[6..9].iter().all(|&x|x>=0));assert!(p[9..12].iter().all(|&x|x>0));assert!(p[16]==2 || p[16]==3);
    let a=&xs[0];let output=&xs[1];assert_eq!(a.shape.len(),5);assert_eq!(output.shape.len(),5);assert_eq!(a.shape[..2],output.shape[..2]);
    assert!(a.shape[2..].iter().all(|&x|x>0));assert_eq!(a.dtype,output.dtype);assert_eq!(xs[2].dtype,4);
    if p[15]!=0 {assert_eq!(xs[2].shape,output.shape);}
    if op==319 {assert_eq!(xs[3].shape,a.shape);assert_eq!(xs[3].dtype,a.dtype);assert!(output.shape[2..].iter().all(|&x|x>0));}
    let c=client();
    match a.dtype {0=>launch::<f32,f32>(&c,op,xs,p),1=>launch::<f16,f32>(&c,op,xs,p),2=>launch::<bf16,f32>(&c,op,xs,p),9=>launch::<f64,f64>(&c,op,xs,p),_=>panic!("pooling requires real floating storage")}
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);
}

#[ruda]
fn index_assert(valid:bool) {
    if !valid {intrinsic!(|scope| {scope.register(ruda_core::ir::Branch::Unreachable);});}
}

#[ruda]
fn fractional_start<F:Float>(sample:F,index:i64,input:i64,output:i64,kernel:i64)->i64 {
    let mut first=input-kernel;
    if index!=output-1 {
        let alpha=F::cast_from(input-kernel)/F::cast_from(output-1);
        first=i64::cast_from((F::cast_from(index)+sample)*alpha)-i64::cast_from(sample*alpha);
    }
    first
}

#[ruda(launch)]
fn fractional_forward<S:Float+RudaElement,F:Float+RudaElement>(a:&Tensor<S>,samples:&Tensor<S>,out:&mut Tensor<S>,indices:&mut Tensor<i64>,kd:i64,kh:i64,kw:i64,#[comptime] rank:u32) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let id=i64::cast_from(a.shape(2));let ih=i64::cast_from(a.shape(3));let iw=i64::cast_from(a.shape(4));
        let od=i64::cast_from(out.shape(2));let oh=i64::cast_from(out.shape(3));let ow=i64::cast_from(out.shape(4));
        let pos=i64::cast_from(p);let w=pos%ow;let h=pos/ow%oh;let d=pos/(ow*oh)%od;let plane=pos/(od*oh*ow);
        let sample_base=usize::cast_from(plane)*rank as usize;
        let mut d0=0i64;let mut w0=0i64;
        if comptime!(rank==2) {
            w0=fractional_start::<F>(F::cast_from(samples[address(samples,sample_base)]),w,iw,ow,kw);
        } else {
            d0=fractional_start::<F>(F::cast_from(samples[address(samples,sample_base)]),d,id,od,kd);
            w0=fractional_start::<F>(F::cast_from(samples[address(samples,sample_base+2)]),w,iw,ow,kw);
        }
        let h0=fractional_start::<F>(F::cast_from(samples[address(samples,sample_base+1)]),h,ih,oh,kh);
        let mut best=F::new(f32::NEG_INFINITY);let mut index=(d0*ih+h0)*iw+w0;
        for z in d0..d0+kd {for y in h0..h0+kh {for x in w0..w0+kw {
            let at=(z*ih+y)*iw+x;let value=F::cast_from(a[address(a,usize::cast_from(plane*id*ih*iw+at))]);
            if value>best || value!=value {best=value;index=at;}
        }}}
        out[address(out,p)]=S::cast_from(best);indices[address(indices,p)]=index;
    }
}

#[ruda(launch)]
fn fractional_backward<S:Float+RudaElement>(grad:&Tensor<S>,indices:&Tensor<i64>,out:&mut Tensor<Atomic<S>>) {
    let p=ABSOLUTE_POS as usize;
    if p<grad.len() {
        let plane=p/(grad.shape(2)*grad.shape(3)*grad.shape(4));let channels=out.shape(1);
        let id=out.shape(2);let ih=out.shape(3);let iw=out.shape(4);let index=indices[address(indices,p)];
        index_assert(index>=0 && index<i64::cast_from(id*ih*iw));
        let spatial=usize::cast_from(index);
        let at=plane/channels*out.stride(0)+plane%channels*out.stride(1)
            +spatial/(ih*iw)*out.stride(2)+spatial/iw%ih*out.stride(3)+spatial%iw*out.stride(4);
        out[at].fetch_add(grad[address(grad,p)]);
    }
}

#[ruda(launch)]
fn unpool_forward<E:Numeric+RudaElement>(a:&Tensor<E>,indices:&Tensor<i64>,out:&mut Tensor<E>) {
    let p=ABSOLUTE_POS as usize;
    if p<a.len() {
        let input_plane=a.shape(2)*a.shape(3)*a.shape(4);let output_plane=out.shape(2)*out.shape(3)*out.shape(4);
        let index=indices[address(indices,p)];index_assert(index>=0 && index<i64::cast_from(output_plane));
        let destination=p/input_plane*output_plane+usize::cast_from(index);
        out[address(out,destination)]=a[address(a,p)];
    }
}

fn launch_fractional<S:Float+RudaElement,F:Float+RudaElement>(c:&ComputeClient<CudaRuntime>,op:u32,xs:&[View],p:&[i64]) {
    let out=&xs[2];if out.len==0 {return;}
    let grid=|n:usize|RudaCount::Static(u32::try_from(n.div_ceil(128)).expect("fractional pooling grid overflow"),1,1);
    let block=RudaDim::new_1d(128);
    unsafe {
        if op==320 {
            fractional_forward::launch::<S,F,CudaRuntime>(c,grid(out.len),block,xs[0].arg(),xs[1].arg(),out.arg(),xs[3].arg(),p[0],p[1],p[2],p[3] as u32);
        } else {
            clear::launch::<S,CudaRuntime>(c,grid(out.len),block,out.arg());
            if xs[0].len>0 {fractional_backward::launch::<S,CudaRuntime>(c,grid(xs[0].len),block,xs[0].arg(),xs[1].arg(),out.arg());}
        }
    }
}

fn launch_unpool<E:Numeric+RudaElement>(c:&ComputeClient<CudaRuntime>,xs:&[View]) {
    let out=&xs[2];if out.len==0 {return;}
    let grid=|n:usize|RudaCount::Static(u32::try_from(n.div_ceil(256)).expect("unpooling grid overflow"),1,1);
    let block=RudaDim::new_1d(256);
    unsafe {
        clear::launch::<E,CudaRuntime>(c,grid(out.len),block,out.arg());
        if xs[0].len>0 {unpool_forward::launch::<E,CudaRuntime>(c,grid(xs[0].len),block,xs[0].arg(),xs[1].arg(),out.arg());}
    }
}

pub(super) fn dispatch_indexed(op:u32,xs:&[View],p:&[i64]) {
    assert!((320..=322).contains(&op));assert_eq!(xs.len(),if op==320 {4}else{3});
    let a=&xs[0];let out=&xs[2];assert_eq!(a.shape.len(),5);assert_eq!(out.shape.len(),5);assert_eq!(a.shape[..2],out.shape[..2]);assert_eq!(a.dtype,out.dtype);
    if op==320 {
        assert_eq!(p.len(),4);assert!(p[3]==2 || p[3]==3);assert_eq!(xs[1].dtype,a.dtype);assert_eq!(xs[1].shape.len(),3);
        assert!(xs[1].shape[0]>=a.shape[0]);assert_eq!(xs[1].shape[1],a.shape[1]);assert_eq!(xs[1].shape[2],p[3] as usize);
        assert_eq!(xs[3].dtype,4);assert_eq!(xs[3].shape,out.shape);
    } else {
        assert_eq!(p.len(),1);assert!(p[0]==2 || p[0]==3);assert_eq!(xs[1].dtype,4);assert_eq!(xs[1].shape,a.shape);
    }
    let c=client();
    if op==322 {
        macro_rules! run {($e:ty)=>{launch_unpool::<$e>(&c,xs)};}
        match a.dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),4=>run!(i64),5=>run!(i32),6=>run!(i16),7=>run!(i8),8=>run!(u8),9=>run!(f64),_=>panic!("unpooling requires real floating or integer storage")}
    } else {
        match a.dtype {0=>launch_fractional::<f32,f32>(&c,op,xs,p),1=>launch_fractional::<f16,f32>(&c,op,xs,p),2=>launch_fractional::<bf16,f32>(&c,op,xs,p),9=>launch_fractional::<f64,f64>(&c,op,xs,p),_=>panic!("fractional pooling requires real floating storage")}
    }
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);
}
