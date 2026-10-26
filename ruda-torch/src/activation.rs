// ATen CUDA activation formulas; retained PyTorch license: bessel.rs.
use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn pointwise<S:Float+RudaElement,F:Float+RudaElement>(a:&Tensor<S>,grad:&Tensor<S>,weight:&Tensor<S>,out:&mut Tensor<S>,p0:F,p1:F,p2:F,#[comptime] op:u32,#[comptime] flag:bool,#[comptime] contiguous:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let mut ai=p;let mut gi=p;let mut wi=p;let mut oi=p;
        if !comptime!(contiguous) {ai=address(a,p);gi=address(grad,p);wi=address(weight,p);oi=address(out,p);}
        out[oi]=S::cast_from(pointwise_value::<S,F>(F::cast_from(a[ai]),F::cast_from(grad[gi]),F::cast_from(weight[wi]),p0,p1,p2,op,flag));
    }
}

#[ruda(launch)]
fn pointwise_vector<S:Float+RudaElement,F:Float+RudaElement>(a:&Array<Vector<S,Const<4>>>,grad:&Array<Vector<S,Const<4>>>,weight:&Array<Vector<S,Const<4>>>,out:&mut Array<Vector<S,Const<4>>>,p0:F,p1:F,p2:F,#[comptime] op:u32,#[comptime] flag:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let x=a[p];let g=grad[p];let w=weight[p];
        let mut result=Vector::<S,Const<4>>::new(S::cast_from(0.0f32));
        #[unroll]
        for lane in 0usize..4 {
            result[lane]=S::cast_from(pointwise_value::<S,F>(F::cast_from(x[lane]),F::cast_from(g[lane]),F::cast_from(w[lane]),p0,p1,p2,op,flag));
        }
        out[p]=result;
    }
}

#[ruda(launch)]
fn initialize_lookup<S:Float+RudaElement>(table:&mut Array<S>,#[comptime] op:u32,#[comptime] flag:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<table.len() {
        let x=S::reinterpret(p as u16);
        table[p]=S::cast_from(pointwise_value::<S,f32>(f32::cast_from(x),0.0f32,0.0f32,0.0f32,0.0f32,0.0f32,op,flag));
    }
}

#[ruda(launch)]
fn pointwise_lookup<S:Float+RudaElement>(a:&Array<Vector<S,Const<4>>>,table:&Array<S>,out:&mut Array<Vector<S,Const<4>>>) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let x=a[p];let mut result=Vector::<S,Const<4>>::new(S::cast_from(0.0f32));
        #[unroll]
        for lane in 0usize..4 {result[lane]=table[u16::reinterpret(x[lane]) as usize];}
        out[p]=result;
    }
}

fn lookup(dtype:u32,op:u32,flag:bool,create:bool)->Option<Handle> {
    thread_local! {
        static TABLES:RefCell<(u64,Vec<(u32,u32,bool,Handle)>)>=const {RefCell::new((0,Vec::new()))};
    }
    TABLES.with(|tables| {
        let mut tables=tables.borrow_mut();let stream=streams::current();
        if tables.0!=stream {*tables=(stream,Vec::new());}
        if let Some((_,_,_,handle))=tables.1.iter().find(|(d,o,f,_)|(*d,*o,*f)==(dtype,op,flag)) {
            return Some(handle.clone());
        }
        if !create {return None;}
        let c=client();let table=View::packed(c.empty(65536*2),vec![65536],dtype);
        let grid=RudaCount::Static(512,1,1);let dim=RudaDim::new_1d(128);
        unsafe {
            if dtype==1 {initialize_lookup::launch::<f16,CudaRuntime>(&c,grid,dim,table.array_arg(),op,flag);}
            else {initialize_lookup::launch::<bf16,CudaRuntime>(&c,grid,dim,table.array_arg(),op,flag);}
        }
        LAUNCHES.fetch_add(1,Ordering::Relaxed);
        tables.1.push((dtype,op,flag,table.handle.clone()));Some(table.handle)
    })
}

#[ruda]
fn pointwise_value<S:Float+RudaElement,F:Float+RudaElement>(x:F,g:F,w:F,p0:F,p1:F,p2:F,#[comptime] op:u32,#[comptime] flag:bool)->F {
        let z=F::new(0.0);let one=F::new(1.0);
        let mut y=x;
        if comptime!(op==265) {
            y=x*p1;if x<=z || x!=x {y=F::cast_from(crate::elementary::expm1(f64::cast_from(x*p2)))*(p0*p1);}
        } else if comptime!(op==266) {
            y=g*p1;if x<=z {
                if comptime!(flag) {y=g*p2*(x+p0*p1);}
                else {y=g*p2*(p0*p1)*(x*p2).exp();}
            }
        } else if comptime!(op==267 || op==268) {
            let mut value=x;if comptime!(op==268) {value=g;}
            y=value*p0;if x>z {y=value;}
        } else if comptime!(op==269 || op==271) {
            let mut clamped=x+F::new(3.0);
            if clamped<z {clamped=z;}if clamped>F::new(6.0) {clamped=F::new(6.0);}
            y=clamped*F::cast_from(1.0f32/6.0f32);
            if comptime!(op==271) {y=x*clamped*F::cast_from(1.0f32/6.0f32);}
        } else if comptime!(op==270) {
            y=z;if x>F::new(-3.0) && x<F::new(3.0) {y=g*F::cast_from(1.0f32/6.0f32);}
        } else if comptime!(op==272) {
            y=g;if x<=F::new(-3.0) {y=z;}else if x<F::new(3.0) {y=g*(x/F::new(3.0)+F::new(0.5));}
        } else if comptime!(op==273) {
            if x*p0>p1 {y=x;}else {y=(x*p0).exp().log1p()/p0;}
        } else if comptime!(op==274) {
            y=g;if x*p0<=p1 || x*p0!=x*p0 {let e=(x*p0).exp();y=g*e/(e+one);}
        } else if comptime!(op==275 || op==276 || op==277) {
            let boundary=F::cast_from(S::cast_from(p0));
            if comptime!(op==275 || op==277) {
                if comptime!(op==277) {y=g;}
                if x>= -boundary && x<=boundary {y=z;}
            } else {
                y=z;if x>boundary {y=x-boundary;}else if x< -boundary {y=x+boundary;}else if x!=x {y=x;}
            }
        } else if comptime!(op==278 || op==279) {
            let t=x.exp().log1p().tanh();y=x*t;
            if comptime!(op==279) {let s=one/(one+(-x).exp());y=g*(t+x*s*(one-t*t));}
        } else if comptime!(op==280 || op==281) {
            let e=(-x.abs()).exp();let mut minimum=z;if x<z {minimum=x;}
            y=minimum-e.log1p();
            if comptime!(op==281) {let mut d=z;let mut sign= -one;if x<z {d=one;sign=one;}y=g*(d-sign*(e/(one+e)));}
        } else if comptime!(op==282 || op==283) {
            let half=F::new(0.5);
            if comptime!(flag) {
                let beta=F::cast_from(0.797884560802865355879892119868764f64);let k=F::cast_from(0.044715f64);
                let sq=x*x;let t=(beta*(x+k*(sq*x))).tanh();y=half*x*(one+t);
                if comptime!(op==283) {y=g*(half*(one+t)+half*x*(one-t*t)*(beta*(one+F::new(3.0)*k*sq)));}
            } else {
                let alpha=F::cast_from(0.707106781186547524400844362104849f64);
                let cdf=half*(one+(x*alpha).erf());y=x*cdf;
                if comptime!(op==283) {let pdf=(-half*x*x).exp()*F::cast_from(0.398942280401432677939946059934382f64);y=g*(cdf+x*pdf);}
            }
        } else if comptime!(op==284 || op==285) {
            let s=one/(one+(-x).exp());y=x/(one+(-x).exp());
            if comptime!(op==285) {y=g*s*(one+x*(one-s));}
        } else if comptime!(op==286) {y=w*x;if x>z {y=x;}}
        else if comptime!(op==287) {y=w*g;if x>z {y=g;}}
        else if comptime!(op==288) {y=x*g;if x>z {y=z;}}
        else if comptime!(op==290) {y=g;if x<=p0 || x>=p1 {y=z;}}
        y
}

#[ruda(launch)]
fn bounded<T:Numeric+RudaElement>(a:&Tensor<T>,other:&Tensor<T>,lower:&Tensor<T>,upper:&Tensor<T>,out:&mut Tensor<T>,#[comptime] clamp:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let x=a[address(a,p)];let lo=lower[0];let hi=upper[0];let mut y=other[address(other,p)];
        if comptime!(clamp) {
            y=x;if y<lo {y=lo;}if y>hi {y=hi;}
            if lo!=lo || hi!=hi {y=lo+hi;}
        } else {if x<=lo {y=hi;}}
        out[address(out,p)]=y;
    }
}

pub(super) fn dispatch(op:u32,xs:&[View],ints:&[i64],values:&[f64]) {
    let bounded=matches!(op,289|291|292);assert_eq!(xs.len(),if bounded {5}else{4});
    let a=&xs[0];let out=xs.last().unwrap();assert_eq!(a.shape,out.shape);
    for (i,x) in xs.iter().enumerate() {assert_eq!(a.dtype,x.dtype);if bounded && (i==2 || i==3) {assert_eq!(x.len,1);}else {assert_eq!(a.shape,x.shape);}}
    if out.len==0 {return;}
    let work=RudaCount::Static(u32::try_from(out.len.div_ceil(128)).expect("activation grid overflow"),1,1);
    let dim=RudaDim::new_1d(128);
    if bounded {
        macro_rules! run {($t:ty)=>{unsafe {bounded::launch::<$t,CudaRuntime>(&client(),work,dim,a.arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),out.arg(),op==289);}}}
        match a.dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),4=>run!(i64),5=>run!(i32),6=>run!(i16),7=>run!(i8),8=>run!(u8),9=>run!(f64),_=>panic!("unsupported activation dtype")}
    } else {
        assert_eq!(values.len(),3);assert_eq!(ints.len(),1);
        let contiguous=xs.iter().all(|view| {
            let mut span=1usize;
            for (&size,&stride) in view.shape.iter().zip(&view.strides).rev() {
                if size>1 && stride!=span {return false;}
                span=span.checked_mul(size).expect("activation stride overflow");
            }
            true
        });
        let vector=contiguous && out.len>=512 && out.len%4==0;
        let vector_work=RudaCount::Static(u32::try_from((out.len/4).div_ceil(128)).expect("activation grid overflow"),1,1);
        if vector && matches!(a.dtype,1|2) && matches!(op,282|284) {
            if let Some(handle)=lookup(a.dtype,op,op==282 && ints[0]!=0,out.len>=65536) {
                let table=View::packed(handle,vec![65536],a.dtype);
                unsafe {
                    if a.dtype==1 {pointwise_lookup::launch::<f16,CudaRuntime>(&client(),vector_work,dim,a.array_arg(),table.array_arg(),out.array_arg());}
                    else {pointwise_lookup::launch::<bf16,CudaRuntime>(&client(),vector_work,dim,a.array_arg(),table.array_arg(),out.array_arg());}
                }
                LAUNCHES.fetch_add(1,Ordering::Relaxed);return;
            }
        }
        macro_rules! run {($s:ty,$f:ty)=>{unsafe {
            if vector {pointwise_vector::launch::<$s,$f,CudaRuntime>(&client(),vector_work,dim,a.array_arg(),xs[1].array_arg(),xs[2].array_arg(),out.array_arg(),values[0] as $f,values[1] as $f,values[2] as $f,op,ints[0]!=0);}
            else {pointwise::launch::<$s,$f,CudaRuntime>(&client(),work,dim,a.arg(),xs[1].arg(),xs[2].arg(),out.arg(),values[0] as $f,values[1] as $f,values[2] as $f,op,ints[0]!=0,contiguous);}
        }}}
        match a.dtype {0=>run!(f32,f32),1=>run!(f16,f32),2=>run!(bf16,f32),9=>run!(f64,f64),_=>panic!("activation requires real floating storage")}
    }
    LAUNCHES.fetch_add(1,Ordering::Relaxed);
}
