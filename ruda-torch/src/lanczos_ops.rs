use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn sinc<F:Float+RudaElement>(x:F)->F {
    let mut result=F::new(1.0);
    if x!=F::new(0.0) {let angle=x*F::cast_from(std::f64::consts::PI);result=angle.sin()/angle;}
    result
}

#[ruda]
fn filter<F:Float+RudaElement>(distance:F)->F {
    let x=distance.abs();let mut value=F::new(0.0);
    if x<F::new(3.0) {value=F::cast_from(f64::cast_from(sinc::<F>(x))*sinc::<f64>(f64::cast_from(x)/3.0));}
    value
}

#[ruda(launch)]
fn weights<F:Float+RudaElement>(out:&mut Tensor<F>,bounds:&mut Tensor<i64>,maxima:&mut Tensor<F>,input_size:usize,scale:F) {
    let p=ABSOLUTE_POS as usize;
    if p<out.shape(0) {
        let capacity=out.shape(1);let center=F::cast_from(f64::cast_from(scale)*(f64::cast_from(p)+0.5));
        let mut support=F::new(3.0);let mut inverse=F::new(1.0);
        if scale>=F::new(1.0) {support=F::cast_from(3.0*f64::cast_from(scale));inverse=F::cast_from(1.0/f64::cast_from(scale));}
        let first=i64::max(i64::cast_from(f64::cast_from(center-support)+0.5),0);
        let last=i64::min(i64::cast_from(f64::cast_from(center+support)+0.5),i64::cast_from(input_size));
        let count=usize::cast_from(i64::min(i64::max(last-first,0),i64::cast_from(capacity)));
        bounds[p*2]=first;bounds[p*2+1]=i64::cast_from(count);
        let base=p*capacity;let mut total=F::new(0.0);let mut maximum=F::new(0.0);
        for j in 0..count {
            let delta=F::cast_from(i64::cast_from(j)+first)-center;
            let distance=F::cast_from((f64::cast_from(delta)+0.5)*f64::cast_from(inverse));
            let value=filter::<F>(distance);out[base+j]=value;total+=value;
        }
        if total!=F::new(0.0) {for j in 0..count {out[base+j]=out[base+j]/total;maximum=F::max(maximum,out[base+j]);}}
        for j in count..capacity {out[base+j]=F::new(0.0);}
        maxima[p]=maximum;
    }
}

#[ruda(launch)]
fn weight_precision(maxima:&Tensor<f64>,precision:&mut Tensor<i32>) {
    if ABSOLUTE_POS==0 {
        let mut maximum=0.0f64;
        for j in 0..maxima.len() {maximum=f64::max(maximum,maxima[j]);}
        let mut bits:u32=0u32;let mut searching=true;
        while bits<22 && searching {
            let next=i32::cast_from(0.5+maximum*f64::cast_from(1u32<<(bits+1)));
            if next>=32768 {searching=false;}else{bits+=1;}
        }
        precision[0]=i32::cast_from(bits);
    }
}

#[ruda(launch)]
fn quantize(weights:&Tensor<f64>,precision:&Tensor<i32>,out:&mut Tensor<i16>) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let scale=f64::cast_from(1u32<<u32::cast_from(precision[0]));let mut value=weights[p]*scale;
        if value<0.0 {value-=0.5;}else{value+=0.5;}
        out[p]=i16::cast_from(i32::cast_from(value));
    }
}

#[ruda]
fn load<S:Numeric+RudaElement>(a:&Tensor<S>,plane:usize,y:usize,x:usize)->S {
    a[address(a,(plane*a.shape(2)+y)*a.shape(3)+x)]
}

#[ruda(launch)]
fn forward_float<F:Float+RudaElement>(a:&Tensor<F>,weights:&Tensor<F>,bounds:&Tensor<i64>,out:&mut Tensor<F>,#[comptime] horizontal:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let x=p%out.shape(3);let y=p/out.shape(3)%out.shape(2);let plane=p/(out.shape(2)*out.shape(3));
        let mut at=y;if comptime!(horizontal) {at=x;}
        let first=usize::cast_from(bounds[at*2]);let count=usize::cast_from(bounds[at*2+1]);let base=at*weights.shape(1);
        let mut value=F::new(0.0);
        for j in 0..count {
            let mut pixel=F::new(0.0);
            if comptime!(horizontal) {pixel=load::<F>(a,plane,y,first+j);}else{pixel=load::<F>(a,plane,first+j,x);}
            let term=pixel*weights[base+j];if j==0 {value=term;}else{value+=term;}
        }
        out[address(out,p)]=value;
    }
}

#[ruda(launch)]
fn forward_byte(a:&Tensor<u8>,weights:&Tensor<i16>,bounds:&Tensor<i64>,precision:&Tensor<i32>,out:&mut Tensor<u8>,#[comptime] horizontal:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let x=p%out.shape(3);let y=p/out.shape(3)%out.shape(2);let plane=p/(out.shape(2)*out.shape(3));
        let mut at=y;if comptime!(horizontal) {at=x;}
        let first=usize::cast_from(bounds[at*2]);let count=usize::cast_from(bounds[at*2+1]);let base=at*weights.shape(1);
        let bits=u32::cast_from(precision[0]);let mut value=i32::cast_from(1u32<<(bits-1));
        for j in 0..count {
            let mut pixel=0u8;
            if comptime!(horizontal) {pixel=load::<u8>(a,plane,y,first+j);}else{pixel=load::<u8>(a,plane,first+j,x);}
            value+=i32::cast_from(pixel)*i32::cast_from(weights[base+j]);
        }
        out[address(out,p)]=u8::cast_from(i32::min(i32::max(value>>i32::cast_from(bits),0),255));
    }
}

#[ruda]
fn contributing_outputs(bounds:&Tensor<i64>,position:i64)->(usize,usize) {
    let mut low=0usize;let mut high=bounds.shape(0);
    while low<high {
        let middle=low+(high-low)/2;let end=bounds[middle*2]+bounds[middle*2+1];
        if end<=position {low=middle+1;}else{high=middle;}
    }
    let first=low;high=bounds.shape(0);
    while low<high {
        let middle=low+(high-low)/2;
        if bounds[middle*2]<=position {low=middle+1;}else{high=middle;}
    }
    (first,low)
}

#[ruda(launch)]
fn backward<F:Float+RudaElement>(grad:&Tensor<F>,wx:&Tensor<F>,bx:&Tensor<i64>,wy:&Tensor<F>,by:&Tensor<i64>,out:&mut Tensor<F>) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let x=p%out.shape(3);let y=p/out.shape(3)%out.shape(2);let plane=p/(out.shape(2)*out.shape(3));
        let (x0,x1)=contributing_outputs(bx,i64::cast_from(x));let (y0,y1)=contributing_outputs(by,i64::cast_from(y));
        let mut value=F::new(0.0);
        for oy in y0..y1 {for ox in x0..x1 {
            let first_x=usize::cast_from(bx[ox*2]);let first_y=usize::cast_from(by[oy*2]);
            let weight=wx[ox*wx.shape(1)+x-first_x]*wy[oy*wy.shape(1)+y-first_y];
            value+=weight*load::<F>(grad,plane,oy,ox);
        }}
        out[address(out,p)]=value;
    }
}

#[ruda(launch)]
fn copy<S:Numeric+RudaElement>(a:&Tensor<S>,out:&mut Tensor<S>) {
    let p=ABSOLUTE_POS as usize;if p<out.len() {out[address(out,p)]=a[address(a,p)];}
}

struct Axis {
    weights:View,
    bounds:View,
    maxima:View,
    precision:View,
    quantized:View,
}

fn allocate(c:&ComputeClient<CudaRuntime>,shape:Vec<usize>,dtype:u32)->View {
    let count=shape.iter().try_fold(1usize,|n,&size|n.checked_mul(size)).expect("Lanczos storage size overflow");
    assert!(count<=u32::MAX as usize,"Lanczos storage exceeds U32 indexing");
    let bytes=count.checked_mul(element_bytes(dtype)).expect("Lanczos storage byte count overflow");
    View::packed(c.empty(bytes),shape,dtype)
}

fn axis(c:&ComputeClient<CudaRuntime>,input:usize,output:usize,scale:f64,dtype:u32,back:bool)->Axis {
    let support=3.0*scale.max(1.0);let support=if dtype==0 {f64::from(support as f32)}else{support};
    let ceiling=if back {(support as f32).ceil() as f64}else{support.ceil()};
    let capacity=(ceiling as usize).saturating_mul(2).saturating_add(1).min(input);
    let weight_dtype=if dtype==8 {9}else{dtype};
    Axis {
        weights:allocate(c,vec![output,capacity],weight_dtype),
        bounds:allocate(c,vec![output,2],4),
        maxima:allocate(c,vec![output],weight_dtype),
        precision:allocate(c,vec![1],5),
        quantized:allocate(c,if dtype==8 {vec![output,capacity]}else{vec![0]},6),
    }
}

fn scale(input:usize,output:usize,supplied:f64,align:bool,dtype:u32)->f64 {
    let wide=dtype!=0;
    let ratio=|n:usize,d:usize|if wide {n as f64/d as f64}else{f64::from(n as f32/d as f32)};
    if align {if output>1 {ratio(input-1,output-1)}else{0.0}}
    else if supplied>0.0 {if wide {1.0/supplied}else{f64::from((1.0/supplied) as f32)}}
    else {ratio(input,output)}
}

fn grid(n:usize)->RudaCount {RudaCount::Static(u32::try_from(n.div_ceil(256)).expect("Lanczos launch overflow"),1,1)}

pub(super) fn dispatch(op:u32,xs:&[View],p:&[i64],s:&[f64]) {
    assert!(op==333 || op==334);assert_eq!(xs.len(),2);assert_eq!(p.len(),1);assert_eq!(s.len(),2);
    assert!(p[0]==0 || p[0]==1);let back=op==334;let a=&xs[0];let out=&xs[1];
    assert_eq!(a.dtype,out.dtype);assert!(a.dtype==0 || a.dtype==9 || !back && a.dtype==8);
    assert_eq!(a.shape.len(),4);assert_eq!(out.shape.len(),4);assert_eq!(a.shape[..2],out.shape[..2]);
    assert!(a.shape[2..].iter().chain(&out.shape[2..]).all(|&n|n>0));
    if out.len==0 {return;}
    let (input,output)=if back {(out,a)}else{(a,out)};let c=client().fixed_execution_queue();let block=RudaDim::new_1d(256);
    let sw=scale(input.shape[3],output.shape[3],s[1],p[0]!=0,a.dtype);
    let sh=scale(input.shape[2],output.shape[2],s[0],p[0]!=0,a.dtype);
    let wx=if back || a.shape[3]!=out.shape[3] {Some(axis(&c,input.shape[3],output.shape[3],sw,a.dtype,back))}else{None};
    let wy=if back || a.shape[2]!=out.shape[2] {Some(axis(&c,input.shape[2],output.shape[2],sh,a.dtype,back))}else{None};
    let temp=if !back && wx.is_some() && wy.is_some() {
        Some(allocate(&c,vec![a.shape[0],a.shape[1],a.shape[2],out.shape[3]],a.dtype))
    }else{None};
    let mut launches=0u64;
    macro_rules! prepare {($f:ty,$table:expr,$size:expr,$scale:expr)=>{{
        let table=$table;
        unsafe {weights::launch::<$f,CudaRuntime>(&c,grid(table.weights.shape[0]),block,table.weights.arg(),table.bounds.arg(),table.maxima.arg(),$size,$scale as $f);}
        launches+=1;
    }};}
    for (table,size,sc) in [(wx.as_ref(),input.shape[3],sw),(wy.as_ref(),input.shape[2],sh)] {
        if let Some(table)=table {
            if a.dtype==0 {prepare!(f32,table,size,sc);}else{prepare!(f64,table,size,sc);}
            if a.dtype==8 {unsafe {
                weight_precision::launch::<CudaRuntime>(&c,RudaCount::Static(1,1,1),RudaDim::new_1d(1),table.maxima.arg(),table.precision.arg());
                quantize::launch::<CudaRuntime>(&c,grid(table.quantized.len),block,table.weights.arg(),table.precision.arg(),table.quantized.arg());
            }launches+=2;}
        }
    }
    macro_rules! run_float {($f:ty)=>{unsafe {
        if back {
            let x=wx.as_ref().unwrap();let y=wy.as_ref().unwrap();
            backward::launch::<$f,CudaRuntime>(&c,grid(out.len),block,a.arg(),x.weights.arg(),x.bounds.arg(),y.weights.arg(),y.bounds.arg(),out.arg());launches+=1;
        } else {
            if let Some(x)=wx.as_ref() {
                let target=temp.as_ref().unwrap_or(out);
                forward_float::launch::<$f,CudaRuntime>(&c,grid(target.len),block,a.arg(),x.weights.arg(),x.bounds.arg(),target.arg(),true);launches+=1;
            }
            if let Some(y)=wy.as_ref() {
                let source=temp.as_ref().unwrap_or(a);
                forward_float::launch::<$f,CudaRuntime>(&c,grid(out.len),block,source.arg(),y.weights.arg(),y.bounds.arg(),out.arg(),false);launches+=1;
            }
            if wx.is_none() && wy.is_none() {copy::launch::<$f,CudaRuntime>(&c,grid(out.len),block,a.arg(),out.arg());launches+=1;}
        }
    }};}
    match a.dtype {
        0=>run_float!(f32),9=>run_float!(f64),
        8=>unsafe {
            if let Some(x)=wx.as_ref() {
                let target=temp.as_ref().unwrap_or(out);
                forward_byte::launch::<CudaRuntime>(&c,grid(target.len),block,a.arg(),x.quantized.arg(),x.bounds.arg(),x.precision.arg(),target.arg(),true);launches+=1;
            }
            if let Some(y)=wy.as_ref() {
                let source=temp.as_ref().unwrap_or(a);
                forward_byte::launch::<CudaRuntime>(&c,grid(out.len),block,source.arg(),y.quantized.arg(),y.bounds.arg(),y.precision.arg(),out.arg(),false);launches+=1;
            }
            if wx.is_none() && wy.is_none() {copy::launch::<u8,CudaRuntime>(&c,grid(out.len),block,a.arg(),out.arg());launches+=1;}
        },
        _=>unreachable!(),
    }
    LAUNCHES.fetch_add(launches,Ordering::Relaxed);finish_dispatch(&c);
}
