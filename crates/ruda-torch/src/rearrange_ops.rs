use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn im2col(a:&Tensor<u8>,out:&mut Tensor<u8>,kh:i64,kw:i64,dh:i64,dw:i64,ph:i64,pw:i64,sh:i64,sw:i64,bh:i64,bw:i64) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let bytes=out.shape(3);let lane=p%bytes;let point=i64::cast_from(p/bytes);
        let block=point%(bh*bw);let kernel_channel=point/(bh*bw)%i64::cast_from(out.shape(1));
        let batch=point/(bh*bw*i64::cast_from(out.shape(1)));let channel=kernel_channel/(kh*kw);
        let y=block/bw*sh-ph+kernel_channel/kw%kh*dh;let x=block%bw*sw-pw+kernel_channel%kw*dw;
        let ih=i64::cast_from(a.shape(2));let iw=i64::cast_from(a.shape(3));let mut value=0u8;
        if y>=0 && y<ih && x>=0 && x<iw {
            let source=usize::cast_from(((batch*i64::cast_from(a.shape(1))+channel)*ih+y)*iw+x)*bytes+lane;
            value=a[address(a,source)];
        }
        out[address(out,p)]=value;
    }
}

#[ruda(launch)]
fn col2im<S:Numeric+RudaElement,F:Numeric+RudaElement>(a:&Tensor<S>,out:&mut Tensor<S>,kh:i64,kw:i64,dh:i64,dw:i64,ph:i64,pw:i64,sh:i64,sw:i64,bh:i64,bw:i64,#[comptime] boolean:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let components=out.shape(4);let lane=p%components;let point=p/components;
        let ih=out.shape(2);let iw=out.shape(3);let plane=point/(ih*iw);
        let y=i64::cast_from(point/iw%ih)+ph;let x=i64::cast_from(point%iw)+pw;
        let eh=(kh-1)*dh+1;let ew=(kw-1)*dw+1;let mut y0=0i64;let mut x0=0i64;
        if y>=eh {y0=(y-eh)/sh+1;}if x>=ew {x0=(x-ew)/sw+1;}
        let y1=i64::min(y/sh+1,bh);let x1=i64::min(x/sw+1,bw);let mut sum=F::cast_from(0u32);
        for by in y0..y1 {for bx in x0..x1 {
            let ky=y-by*sh;let kx=x-bx*sw;
            if ky%dh==0 && kx%dw==0 {
                let source=usize::cast_from(((i64::cast_from(plane)*kh+ky/dh)*kw+kx/dw)*bh*bw+by*bw+bx)*components+lane;
                let value=F::cast_from(a[address(a,source)]);
                if comptime!(boolean) {sum=F::cast_from(sum!=F::cast_from(0u32) || value!=F::cast_from(0u32));}
                else {sum+=value;}
            }
        }}
        out[address(out,p)]=S::cast_from(sum);
    }
}

#[ruda(launch)]
fn shuffle(a:&Tensor<u8>,out:&mut Tensor<u8>,factor:usize,#[comptime] mode:u32) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let bytes=out.shape(out.rank()-1);let lane=p%bytes;let point=p/bytes;let mut source=0usize;
        if comptime!(mode==2) {
            let channels=out.shape(1);let mut spatial=1usize;
            for axis in 2..out.rank()-1 {spatial*=out.shape(axis);}
            let channel=point/spatial%channels;let batch=point/(spatial*channels);
            let mapped=channel%factor*(channels/factor)+channel/factor;
            source=(batch*channels+mapped)*spatial+point%spatial;
        } else {
            let rank=out.rank();let oc=out.shape(rank-4);let oh=out.shape(rank-3);let ow=out.shape(rank-2);
            let ic=a.shape(rank-4);let ih=a.shape(rank-3);let iw=a.shape(rank-2);
            let x=point%ow;let y=point/ow%oh;let channel=point/(ow*oh)%oc;let batch=point/(ow*oh*oc);
            let mut sx=0usize;let mut sy=0usize;let mut sc=0usize;
            if comptime!(mode==0) {sx=x/factor;sy=y/factor;sc=(channel*factor+y%factor)*factor+x%factor;}
            else {sx=x*factor+channel%factor;sy=y*factor+channel/factor%factor;sc=channel/(factor*factor);}
            source=((batch*ic+sc)*ih+sy)*iw+sx;
        }
        out[address(out,p)]=a[address(a,source*bytes+lane)];
    }
}

#[ruda(launch)]
fn unfold_backward<S:Numeric+RudaElement>(grad:&Tensor<S>,out:&mut Tensor<S>,dim:u32,size:i64,step:i64,#[comptime] boolean:bool,#[comptime] copy_only:bool) {
    let dim = dim as usize;
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let components=out.shape(out.rank()-1);let lane=p%components;let mut point=p/components;
        let mut base=lane*grad.stride(grad.rank()-1);let mut position=0i64;
        for axis in 0..out.rank()-1 {
            let d=out.rank()-2-axis;let coordinate=point%out.shape(d);point/=out.shape(d);
            if d==dim {position=i64::cast_from(coordinate);}else{base+=coordinate*grad.stride(d);}
        }
        let mut first=0i64;if position>size {first=(position-size)/step;}
        if !(first*step<=position && position<first*step+size) {first+=1;}
        let last=i64::min(position/step,i64::cast_from(grad.shape(dim))-1);let mut sum=S::cast_from(0u32);
        for fold in first..last+1 {
            let at=base+usize::cast_from(fold)*grad.stride(dim)+usize::cast_from(position-fold*step)*grad.stride(grad.rank()-2);
            let value=grad[at];
            if comptime!(copy_only) {sum=value;}
            else if comptime!(boolean) {sum=S::cast_from(sum!=S::cast_from(0u32) || value!=S::cast_from(0u32));}
            else {sum+=value;}
        }
        out[address(out,p)]=sum;
    }
}

fn indexable(v:&View) {
    assert!(v.len<=u32::MAX as usize && v.shape.iter().all(|&n|n<=u32::MAX as usize),"rearrangement shape exceeds U32 indexing");
    if v.len!=0 {assert!(v.shape.iter().zip(&v.strides).try_fold(1usize,|span,(&n,&s)|
        (n-1).checked_mul(s).and_then(|term|span.checked_add(term))).is_some_and(|span|span<=u32::MAX as usize),
        "rearrangement strided span exceeds U32 indexing");}
}

fn components(v:&View)->View {
    let width=if v.dtype==10 || v.dtype==11 {2usize}else{1};let mut shape=v.shape.clone();shape.push(width);
    let mut strides:Vec<usize>=v.strides.iter().map(|s|s.checked_mul(width).expect("component stride overflow")).collect();strides.push(1);
    let result=View {handle:v.handle.clone(),shape,strides,len:v.len.checked_mul(width).expect("component count overflow"),
        dtype:match v.dtype {10=>0,11=>9,3|12|13=>8,dtype=>dtype}};
    indexable(&result);result
}

fn grid(n:usize)->RudaCount {RudaCount::Static(u32::try_from(n.div_ceil(256)).expect("rearrangement launch overflow"),1,1)}

#[ruda(launch)]
fn roll_bytes(a:&Tensor<u8>,out:&mut Tensor<u8>,#[comptime] shifts:Vec<usize>,#[comptime] flatten:bool,#[comptime] rank:usize) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let bytes=out.shape(rank);let lane=p%bytes;let mut point=p/bytes;let mut source=lane;
        if comptime!(flatten) {
            let shift=comptime!(shifts[0]);let length=out.len()/bytes;
            if point<shift {point+=length-shift;}else {point-=shift;}
            source=address(a,point*bytes+lane);
        } else {
            #[unroll]
            for axis in 0..rank {
                let d=comptime!(rank-axis-1);let size=a.shape(d);let mut coordinate=point%size;point/=size;
                let shift=comptime!(shifts[d]);
                if coordinate<shift {coordinate+=size-shift;}else {coordinate-=shift;}
                source+=coordinate*a.stride(d);
            }
        }
        out[address(out,p)]=a[source];
    }
}

pub(super) fn roll_dispatch(xs:&[View],ints:&[i64]) {
    assert_eq!(xs.len(),2);assert!(!ints.is_empty() && matches!(ints[0],0|1));
    let a=&xs[0];let out=&xs[1];assert_eq!(a.dtype,out.dtype);assert_eq!(a.shape,out.shape);
    assert!(a.dtype<=11 || a.dtype==14);
    let flatten=ints[0]!=0;let rank=a.shape.len();
    assert_eq!(ints.len(),1+if flatten {1}else{rank});
    let mut shifts=Vec::with_capacity(ints.len()-1);
    for (axis,&shift) in ints[1..].iter().enumerate() {
        let size=if flatten {a.len}else{a.shape[axis]};
        let value=usize::try_from(shift).expect("negative normalized roll shift");
        assert!(value<size.max(1));shifts.push(value);
    }
    if out.len==0 {return;}
    let a=a.byte_view();let out=out.byte_view();indexable(&a);indexable(&out);
    let c=client();
    unsafe {roll_bytes::launch::<CudaRuntime>(&c,grid(out.len),RudaDim::new_1d(256),a.arg(),out.arg(),shifts,flatten,rank);}
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);
}

fn windows(input:usize,kernel:i64,dilation:i64,padding:i64,stride:i64)->usize {
    let extent=dilation.checked_mul(kernel-1).and_then(|v|v.checked_add(1)).expect("window extent overflow");
    let padded=padding.checked_mul(2).and_then(|v|v.checked_add(i64::try_from(input).unwrap())).expect("padded window size overflow");
    assert!(padded>=extent,"window count must be positive");
    usize::try_from((padded-extent)/stride+1).expect("window count overflow")
}

pub(super) fn dispatch(op:u32,xs:&[View],p:&[i64]) {
    assert!((335..=338).contains(&op));assert_eq!(xs.len(),2);let a=&xs[0];let out=&xs[1];assert_eq!(a.dtype,out.dtype);
    assert!(a.dtype<=13);let c=client();let block=RudaDim::new_1d(256);
    if op==335 || op==336 {
        assert_eq!(p.len(),8);assert!(p[..4].iter().chain(&p[6..]).all(|&v|v>0));assert!(p[4]>=0 && p[5]>=0);
        assert!(matches!(a.dtype,0|1|2|3|9|10|11));
        let (image,columns)=if op==335 {(a,out)}else{(out,a)};
        assert_eq!(image.shape.len(),4);assert_eq!(columns.shape.len(),3);assert_eq!(image.shape[0],columns.shape[0]);
        assert!(image.shape[1..].iter().all(|&n|n>0));
        let bh=windows(image.shape[2],p[0],p[2],p[4],p[6]);let bw=windows(image.shape[3],p[1],p[3],p[5],p[7]);
        let kernel=(p[0] as usize).checked_mul(p[1] as usize).expect("kernel area overflow");
        assert_eq!(columns.shape[1],image.shape[1].checked_mul(kernel).expect("column channel count overflow"));
        assert_eq!(columns.shape[2],bh.checked_mul(bw).expect("column position count overflow"));
        if out.len==0 {return;}
        if op==335 {
            let a=a.byte_view();let out=out.byte_view();indexable(&a);indexable(&out);
            unsafe {im2col::launch::<CudaRuntime>(&c,grid(out.len),block,a.arg(),out.arg(),p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],bh as i64,bw as i64);}
        } else {
            let boolean=a.dtype==3;let a=components(a);let out=components(out);
            macro_rules! run {($s:ty,$f:ty)=>{unsafe {col2im::launch::<$s,$f,CudaRuntime>(&c,grid(out.len),block,a.arg(),out.arg(),p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],bh as i64,bw as i64,boolean);}};}
            match a.dtype {0=>run!(f32,f32),1=>run!(f16,f32),2=>run!(bf16,f32),8=>run!(u8,u8),9=>run!(f64,f64),_=>unreachable!()}
        }
    } else if op==337 {
        assert_eq!(p.len(),2);assert!((0..=2).contains(&p[0]));assert!(p[1]>0);let factor=p[1] as usize;
        assert_eq!(a.shape.len(),out.shape.len());assert!(a.shape.len()>=3);assert_eq!(a.len,out.len);
        if p[0]==2 {assert_eq!(a.shape,out.shape);assert_eq!(a.shape[1]%factor,0);}
        else {
            let rank=a.shape.len();assert_eq!(a.shape[..rank-3],out.shape[..rank-3]);
            let square=factor.checked_mul(factor).expect("pixel shuffle factor overflow");
            let (small,large)=if p[0]==0 {(a,out)}else{(out,a)};
            assert_eq!(small.shape[rank-3],large.shape[rank-3].checked_mul(square).expect("pixel shuffle channel overflow"));
            for d in rank-2..rank {assert_eq!(large.shape[d],small.shape[d].checked_mul(factor).expect("pixel shuffle spatial overflow"));}
        }
        if out.len==0 {return;}
        let a=a.byte_view();let out=out.byte_view();indexable(&a);indexable(&out);
        unsafe {shuffle::launch::<CudaRuntime>(&c,grid(out.len),block,a.arg(),out.arg(),factor,p[0] as u32);}
    } else {
        assert_eq!(p.len(),4);assert!(p[0]>=0 && (p[0] as usize)<out.shape.len());assert!(p[1]>=0 && p[2]>0);assert!(p[3]==0 || p[3]==1);
        let dim=p[0] as usize;let size=p[1] as usize;let step=p[2] as usize;assert!(size<=out.shape[dim]);
        let mut expected=out.shape.clone();expected[dim]=if p[3]!=0 {1}else{(expected[dim]-size)/step+1};expected.push(size);
        assert_eq!(a.shape,expected);assert!(a.dtype<12 || step>=size);
        if out.len==0 {return;}
        let boolean=a.dtype==3;let a=components(a);let out=components(out);
        macro_rules! run {($s:ty)=>{unsafe {unfold_backward::launch::<$s,CudaRuntime>(&c,grid(out.len),block,a.arg(),out.arg(),dim as u32,p[1],p[2],boolean,step>=size);}};}
        match a.dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),4=>run!(i64),5=>run!(i32),6=>run!(i16),7=>run!(i8),8=>run!(u8),9=>run!(f64),_=>unreachable!()}
    }
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);
}
