use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn nearest_index(position:usize,size:usize,scale:f32,#[comptime] exact:bool,#[comptime] backward:bool)->usize {
    let mut x=f32::cast_from(position);
    if comptime!(backward) {
        x*=scale;
        if comptime!(exact) {x-=0.5;}
        usize::cast_from(i64::min(i64::cast_from(x.ceil()),i64::cast_from(size)))
    } else {
        if comptime!(exact) {x+=0.5;}
        x*=scale;
        usize::cast_from(i64::min(i64::cast_from(x.floor()),i64::cast_from(size)-1))
    }
}

#[ruda(launch)]
fn nearest_forward<S:Numeric+RudaElement>(a:&Tensor<S>,out:&mut Tensor<S>,sd:f32,sh:f32,sw:f32,#[comptime] exact:bool,#[comptime] rank:u32) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let ow=out.shape(4);let oh=out.shape(3);let od=out.shape(2);
        let w=p%ow;let h=p/ow%oh;let d=p/(ow*oh)%od;let plane=p/(ow*oh*od);
        let iw=a.shape(4);let ih=a.shape(3);let id=a.shape(2);
        let mut x=nearest_index(w,iw,sw,exact,false);let mut y=nearest_index(h,ih,sh,exact,false);
        let z=nearest_index(d,id,sd,exact,false);
        if comptime!(rank==2) {if iw==ow {x=w;}if ih==oh {y=h;}}
        let source=((plane*id+z)*ih+y)*iw+x;
        out[address(out,p)]=a[address(a,source)];
    }
}

#[ruda(launch)]
fn nearest_backward<S:Numeric+RudaElement,F:Numeric+RudaElement>(grad:&Tensor<S>,out:&mut Tensor<S>,sd:f32,sh:f32,sw:f32,#[comptime] exact:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let iw=out.shape(4);let ih=out.shape(3);let id=out.shape(2);
        let w=p%iw;let h=p/iw%ih;let d=p/(iw*ih)%id;let plane=p/(iw*ih*id);
        let ow=grad.shape(4);let oh=grad.shape(3);let od=grad.shape(2);
        let w0=nearest_index(w,ow,sw,exact,true);let w1=nearest_index(w+1,ow,sw,exact,true);
        let h0=nearest_index(h,oh,sh,exact,true);let h1=nearest_index(h+1,oh,sh,exact,true);
        let d0=nearest_index(d,od,sd,exact,true);let d1=nearest_index(d+1,od,sd,exact,true);
        let mut sum=F::cast_from(0u32);
        for z in d0..d1 {for y in h0..h1 {for x in w0..w1 {
            sum+=F::cast_from(grad[address(grad,((plane*od+z)*oh+y)*ow+x)]);
        }}}
        out[address(out,p)]=S::cast_from(sum);
    }
}

#[ruda]
fn coordinate<F:Float+RudaElement>(position:usize,scale:F,#[comptime] align:bool,#[comptime] cubic:bool)->F {
    let mut x=scale*F::cast_from(position);
    if comptime!(!align) {
        x=scale*(F::cast_from(position)+F::new(0.5))-F::new(0.5);
        if comptime!(!cubic) {if x<F::new(0.0) {x=F::new(0.0);}}
    }
    x
}

#[ruda]
fn linear_coordinate<F:Float+RudaElement>(position:usize,size:usize,scale:F,#[comptime] align:bool)->(usize,usize,F,F) {
    let x=coordinate(position,scale,align,false);let first=usize::cast_from(x);let mut last=first;
    if first<size-1 {last+=1;}
    let weight=x-F::cast_from(first);
    (first,last,F::new(1.0)-weight,weight)
}

#[ruda]
fn load<S:Float+RudaElement,F:Float+RudaElement>(a:&Tensor<S>,plane:usize,z:usize,y:usize,x:usize)->F {
    F::cast_from(a[address(a,((plane*a.shape(2)+z)*a.shape(3)+y)*a.shape(4)+x)])
}

#[ruda]
fn cubic_coefficient<F:Float+RudaElement>(t:F,index:usize)->F {
    let mut x=t;let mut result=F::new(0.0);let a=F::new(-0.75);
    if index>=2 {x=F::new(1.0)-t;}
    if index==0 || index==3 {
        x+=F::new(1.0);result=((a*x-F::new(5.0)*a)*x+F::new(8.0)*a)*x-F::new(4.0)*a;
    } else {result=((a+F::new(2.0))*x-(a+F::new(3.0)))*x*x+F::new(1.0);}
    result
}

#[ruda(launch)]
fn weighted_forward<S:Float+RudaElement,F:Float+RudaElement>(a:&Tensor<S>,out:&mut Tensor<S>,sd:F,sh:F,sw:F,#[comptime] align:bool,#[comptime] rank:u32,#[comptime] cubic:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let ow=out.shape(4);let oh=out.shape(3);let od=out.shape(2);let plane=p/(ow*oh*od);
        let w=p%ow;let h=p/ow%oh;let d=p/(ow*oh)%od;
        let iw=a.shape(4);let ih=a.shape(3);let id=a.shape(2);let mut result=F::new(0.0);
        if comptime!(cubic) {
            let rx=coordinate(w,sw,align,true);let ry=coordinate(h,sh,align,true);
            let x=i64::cast_from(f32::cast_from(rx).floor());let y=i64::cast_from(f32::cast_from(ry).floor());
            let tx=rx-F::cast_from(x);let ty=ry-F::cast_from(y);let mut rows=Array::<F>::new(4);
            let c0=cubic_coefficient(tx,0);let c1=cubic_coefficient(tx,1);let c2=cubic_coefficient(tx,2);let c3=cubic_coefficient(tx,3);
            let x0=usize::cast_from(i64::min(i64::max(x-1,0),i64::cast_from(iw)-1));
            let x1=usize::cast_from(i64::min(i64::max(x,0),i64::cast_from(iw)-1));
            let x2=usize::cast_from(i64::min(i64::max(x+1,0),i64::cast_from(iw)-1));
            let x3=usize::cast_from(i64::min(i64::max(x+2,0),i64::cast_from(iw)-1));
            for k in 0..4usize {
                let yy=usize::cast_from(i64::min(i64::max(y-1+i64::cast_from(k),0),i64::cast_from(ih)-1));
                rows[k]=load::<S,F>(a,plane,0,yy,x0)*c0+load::<S,F>(a,plane,0,yy,x1)*c1
                    +load::<S,F>(a,plane,0,yy,x2)*c2+load::<S,F>(a,plane,0,yy,x3)*c3;
            }
            result=rows[0]*cubic_coefficient(ty,0)+rows[1]*cubic_coefficient(ty,1)
                +rows[2]*cubic_coefficient(ty,2)+rows[3]*cubic_coefficient(ty,3);
        } else {
            let (x0,x1,wx0,wx1)=linear_coordinate(w,iw,sw,align);
            if comptime!(rank==1) {result=wx0*load::<S,F>(a,plane,0,0,x0)+wx1*load::<S,F>(a,plane,0,0,x1);}
            else {
                let (y0,y1,wy0,wy1)=linear_coordinate(h,ih,sh,align);
                let (z0,z1,wz0,wz1)=linear_coordinate(d,id,sd,align);
                let top=wy0*(wx0*load::<S,F>(a,plane,z0,y0,x0)+wx1*load::<S,F>(a,plane,z0,y0,x1))
                    +wy1*(wx0*load::<S,F>(a,plane,z0,y1,x0)+wx1*load::<S,F>(a,plane,z0,y1,x1));
                result=top;
                if comptime!(rank==3) {
                    let bottom=wy0*(wx0*load::<S,F>(a,plane,z1,y0,x0)+wx1*load::<S,F>(a,plane,z1,y0,x1))
                        +wy1*(wx0*load::<S,F>(a,plane,z1,y1,x0)+wx1*load::<S,F>(a,plane,z1,y1,x1));
                    result=wz0*top+wz1*bottom;
                }
            }
        }
        out[address(out,p)]=S::cast_from(result);
    }
}

#[ruda]
fn add<S:Float+RudaElement,F:Float+RudaElement>(out:&mut Tensor<Atomic<S>>,plane:usize,z:usize,y:usize,x:usize,value:F) {
    let channels=out.shape(1);
    let at=plane/channels*out.stride(0)+plane%channels*out.stride(1)+z*out.stride(2)+y*out.stride(3)+x*out.stride(4);
    out[at].fetch_add(S::cast_from(value));
}

#[ruda(launch)]
fn weighted_backward<S:Float+RudaElement,F:Float+RudaElement>(grad:&Tensor<S>,out:&mut Tensor<Atomic<S>>,sd:F,sh:F,sw:F,#[comptime] align:bool,#[comptime] rank:u32,#[comptime] cubic:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<grad.len() {
        let ow=grad.shape(4);let oh=grad.shape(3);let od=grad.shape(2);let plane=p/(ow*oh*od);
        let w=p%ow;let h=p/ow%oh;let d=p/(ow*oh)%od;
        let iw=out.shape(4);let ih=out.shape(3);let id=out.shape(2);let value=F::cast_from(grad[address(grad,p)]);
        if comptime!(cubic) {
            let rx=coordinate(w,sw,align,true);let ry=coordinate(h,sh,align,true);
            let x=i64::cast_from(f32::cast_from(rx).floor());let y=i64::cast_from(f32::cast_from(ry).floor());
            let tx=rx-F::cast_from(x);let ty=ry-F::cast_from(y);
            for ky in 0..4usize {for kx in 0..4usize {
                let yy=usize::cast_from(i64::min(i64::max(y-1+i64::cast_from(ky),0),i64::cast_from(ih)-1));
                let xx=usize::cast_from(i64::min(i64::max(x-1+i64::cast_from(kx),0),i64::cast_from(iw)-1));
                add::<S,F>(out,plane,0,yy,xx,value*cubic_coefficient(ty,ky)*cubic_coefficient(tx,kx));
            }}
        } else {
            let (x0,x1,wx0,wx1)=linear_coordinate(w,iw,sw,align);
            if comptime!(rank==1) {add::<S,F>(out,plane,0,0,x0,wx0*value);add::<S,F>(out,plane,0,0,x1,wx1*value);}
            else {
                let (y0,y1,wy0,wy1)=linear_coordinate(h,ih,sh,align);
                let (z0,z1,wz0,wz1)=linear_coordinate(d,id,sd,align);
                if comptime!(rank==2) {
                    add::<S,F>(out,plane,0,y0,x0,wy0*wx0*value);add::<S,F>(out,plane,0,y0,x1,wy0*wx1*value);
                    add::<S,F>(out,plane,0,y1,x0,wy1*wx0*value);add::<S,F>(out,plane,0,y1,x1,wy1*wx1*value);
                } else {
                    add::<S,F>(out,plane,z0,y0,x0,wz0*wy0*wx0*value);add::<S,F>(out,plane,z0,y0,x1,wz0*wy0*wx1*value);
                    add::<S,F>(out,plane,z0,y1,x0,wz0*wy1*wx0*value);add::<S,F>(out,plane,z0,y1,x1,wz0*wy1*wx1*value);
                    add::<S,F>(out,plane,z1,y0,x0,wz1*wy0*wx0*value);add::<S,F>(out,plane,z1,y0,x1,wz1*wy0*wx1*value);
                    add::<S,F>(out,plane,z1,y1,x0,wz1*wy1*wx0*value);add::<S,F>(out,plane,z1,y1,x1,wz1*wy1*wx1*value);
                }
            }
        }
    }
}

#[ruda(launch)]
fn clear<S:Float+RudaElement>(out:&mut Tensor<S>) {
    let p=ABSOLUTE_POS as usize;if p<out.len() {out[address(out,p)]=S::new(0.0);}
}

#[ruda(launch)]
fn copy<S:Numeric+RudaElement>(a:&Tensor<S>,out:&mut Tensor<S>) {
    let p=ABSOLUTE_POS as usize;if p<out.len() {out[address(out,p)]=a[address(a,p)];}
}

#[ruda]
fn aa_assert(valid:bool) {
    if !valid {intrinsic!(|scope| {scope.register(ruda_core::ir::Branch::Unreachable);});}
}

#[ruda]
fn aa_filter<F:Float+RudaElement>(distance:F,#[comptime] cubic:bool)->F {
    let mut x=distance;if x<F::new(0.0) {x=-x;}
    let mut weight=F::new(0.0);
    if comptime!(cubic) {
        let a=F::new(-0.5);
        if x<F::new(1.0) {weight=((a+F::new(2.0))*x-(a+F::new(3.0)))*x*x+F::new(1.0);}
        else if x<F::new(2.0) {weight=(((x-F::new(5.0))*x+F::new(8.0))*x-F::new(4.0))*a;}
    } else {if x<F::new(1.0) {weight=F::new(1.0)-x;}}
    weight
}

#[ruda]
fn aa_span<F:Float+RudaElement>(position:usize,size:usize,scale:F,#[comptime] cubic:bool)->(usize,usize,F) {
    let mut support=F::new(1.0);if comptime!(cubic) {support=F::new(2.0);}
    if scale>=F::new(1.0) {support*=scale;}
    let center=scale*(F::cast_from(position)+F::new(0.5));
    let first=i64::max(i64::cast_from(center-support+F::new(0.5)),0);
    let last=i64::min(i64::cast_from(center+support+F::new(0.5)),i64::cast_from(size));
    aa_assert(first<last);
    (usize::cast_from(first),usize::cast_from(last-first),center)
}

#[ruda(launch)]
fn aa_weights<S:Float+RudaElement,F:Float+RudaElement>(weights:&mut Tensor<S>,input_size:usize,scale:F,#[comptime] cubic:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<weights.shape(0) {
        let (first,count,center)=aa_span(p,input_size,scale,cubic);let capacity=weights.shape(1);
        aa_assert(count<=capacity);
        let mut inverse=F::new(1.0);if scale>=F::new(1.0) {inverse=F::new(1.0)/scale;}
        let offset=F::cast_from(first)-center;let mut total=F::new(0.0);
        for j in 0..count {
            let value=aa_filter((F::cast_from(j)+offset+F::new(0.5))*inverse,cubic);
            weights[p*capacity+j]=S::cast_from(value);total+=value;
        }
        if total!=F::new(0.0) {
            let denominator=S::cast_from(total);
            for j in 0..count {weights[p*capacity+j]=weights[p*capacity+j]/denominator;}
        }
        for j in count..capacity {weights[p*capacity+j]=S::new(0.0);}
    }
}

#[ruda(launch)]
fn aa_forward<S:Float+RudaElement,F:Float+RudaElement>(a:&Tensor<S>,wx:&Tensor<S>,wy:&Tensor<S>,out:&mut Tensor<S>,sh:F,sw:F,#[comptime] cubic:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let ow=out.shape(4);let oh=out.shape(3);let x=p%ow;let y=p/ow%oh;let plane=p/(ow*oh);
        let (first_x,count_x,_center_x)=aa_span(x,a.shape(4),sw,cubic);
        let (first_y,count_y,_center_y)=aa_span(y,a.shape(3),sh,cubic);
        let xbase=x*wx.shape(1);let ybase=y*wy.shape(1);let mut total=F::new(0.0);
        for row in 0..count_y {
            let mut value=load::<S,F>(a,plane,0,first_y+row,first_x)*F::cast_from(wx[xbase]);
            for column in 1..count_x {
                value+=load::<S,F>(a,plane,0,first_y+row,first_x+column)*F::cast_from(wx[xbase+column]);
            }
            let rounded=S::cast_from(value);let term=F::cast_from(rounded)*F::cast_from(wy[ybase+row]);
            if row==0 {total=term;}else{total+=term;}
        }
        out[address(out,p)]=S::cast_from(total);
    }
}

#[ruda(launch)]
fn aa_backward<S:Float+RudaElement,F:Float+RudaElement>(grad:&Tensor<S>,wx:&Tensor<S>,wy:&Tensor<S>,out:&mut Tensor<Atomic<S>>,sh:F,sw:F,#[comptime] cubic:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<grad.len() {
        let ow=grad.shape(4);let oh=grad.shape(3);let x=p%ow;let y=p/ow%oh;let plane=p/(ow*oh);
        let (first_x,count_x,_center_x)=aa_span(x,out.shape(4),sw,cubic);
        let (first_y,count_y,_center_y)=aa_span(y,out.shape(3),sh,cubic);
        let xbase=x*wx.shape(1);let ybase=y*wy.shape(1);let value=grad[address(grad,p)];
        for row in 0..count_y {for column in 0..count_x {
            let weight=wx[xbase+column]*wy[ybase+row];let contribution=weight*value;
            add::<S,F>(out,plane,0,first_y+row,first_x+column,F::cast_from(contribution));
        }}
    }
}

fn aa_table(c:&ComputeClient<CudaRuntime>,input:usize,output:usize,scale:f64,dtype:u32,cubic:bool)->View {
    let scale=if dtype==9 {scale}else{f64::from(scale as f32)};
    let support=(if cubic {2.0}else{1.0})*scale.max(1.0);
    let capacity=((support as f32).ceil() as usize).saturating_mul(2).saturating_add(1).min(input);
    let count=output.checked_mul(capacity).expect("antialias weight count overflow");
    assert!(count<=u32::MAX as usize,"antialias weights exceed U32 indexing");
    let bytes=count.checked_mul(element_bytes(dtype)).expect("antialias weight storage overflow");
    View::packed(c.empty(bytes),vec![output,capacity],dtype)
}

fn scales(input:&View,output:&View,p:&[i64],s:&[f64],backward:bool,nearest:bool)->[f64;3] {
    let wide=!nearest && input.dtype==9;
    let ratio=|n:usize,d:usize|if wide {n as f64/d as f64}else{f64::from(n as f32/d as f32)};
    std::array::from_fn(|axis| {
        let n=input.shape[axis+2];let m=output.shape[axis+2];
        if nearest && backward {if s[axis]>0.0 {s[axis]}else{ratio(m,n)}}
        else if !nearest && p[2]!=0 {if m>1 {ratio(n-1,m-1)}else{0.0}}
        else if s[axis]>0.0 {1.0/s[axis]}else{ratio(n,m)}
    })
}

fn grid(n:usize)->RudaCount {RudaCount::Static(u32::try_from(n.div_ceil(256)).expect("resize grid overflow"),1,1)}

pub(super) fn dispatch(op:u32,xs:&[View],p:&[i64],s:&[f64]) {
    assert!((325..=330).contains(&op));assert_eq!(xs.len(),2);assert_eq!(p.len(),3);assert_eq!(s.len(),3);
    assert!((1..=3).contains(&p[0]));assert!((0..=5).contains(&p[1]));assert!(p[2]==0 || p[2]==1);
    let nearest=op<=326;let antialias=op>=329;let backward=op==326 || op==328 || op==330;
    assert_eq!(nearest,p[1]<2);assert_eq!(antialias,p[1]>=4);assert!(p[1]<3 || p[0]==2);
    let a=&xs[0];let out=&xs[1];assert_eq!(a.dtype,out.dtype);assert_eq!(a.shape.len(),5);assert_eq!(out.shape.len(),5);
    assert_eq!(a.shape[..2],out.shape[..2]);assert!(a.shape[2..].iter().chain(&out.shape[2..]).all(|&n|n>0));
    for axis in 2..5-p[0] as usize {assert_eq!(a.shape[axis],1);assert_eq!(out.shape[axis],1);}
    let (input,output)=if backward {(out,a)}else{(a,out)};
    let sc=scales(input,output,p,s,backward,nearest);
    let identity=a.shape==out.shape && (if antialias {backward}else{!nearest || p[0]==2});
    if out.len==0 {return;}
    let c=if antialias {client().fixed_execution_queue()}else{client()};let block=RudaDim::new_1d(256);
    let tables=if antialias && !identity {
        Some((aa_table(&c,input.shape[4],output.shape[4],sc[2],a.dtype,p[1]==5),
              aa_table(&c,input.shape[3],output.shape[3],sc[1],a.dtype,p[1]==5)))
    }else{None};
    macro_rules! run_nearest {($storage:ty,$acc:ty)=>{unsafe {
        if identity {copy::launch::<$storage,CudaRuntime>(&c,grid(out.len),block,a.arg(),out.arg());}
        else if backward {nearest_backward::launch::<$storage,$acc,CudaRuntime>(&c,grid(out.len),block,a.arg(),out.arg(),sc[0] as f32,sc[1] as f32,sc[2] as f32,p[1]==1);}
        else {nearest_forward::launch::<$storage,CudaRuntime>(&c,grid(out.len),block,a.arg(),out.arg(),sc[0] as f32,sc[1] as f32,sc[2] as f32,p[1]==1,p[0] as u32);}
    }};}
    macro_rules! run_weighted {($storage:ty,$acc:ty)=>{unsafe {
        if identity {copy::launch::<$storage,CudaRuntime>(&c,grid(out.len),block,a.arg(),out.arg());}
        else if antialias {
            let (wx,wy)=tables.as_ref().unwrap();
            aa_weights::launch::<$storage,$acc,CudaRuntime>(&c,grid(output.shape[4]),block,wx.arg(),input.shape[4],sc[2] as $acc,p[1]==5);
            aa_weights::launch::<$storage,$acc,CudaRuntime>(&c,grid(output.shape[3]),block,wy.arg(),input.shape[3],sc[1] as $acc,p[1]==5);
            if backward {
                clear::launch::<$storage,CudaRuntime>(&c,grid(out.len),block,out.arg());
                aa_backward::launch::<$storage,$acc,CudaRuntime>(&c,grid(a.len),block,a.arg(),wx.arg(),wy.arg(),out.arg(),sc[1] as $acc,sc[2] as $acc,p[1]==5);
            }else{aa_forward::launch::<$storage,$acc,CudaRuntime>(&c,grid(out.len),block,a.arg(),wx.arg(),wy.arg(),out.arg(),sc[1] as $acc,sc[2] as $acc,p[1]==5);}
        }
        else if backward {
            clear::launch::<$storage,CudaRuntime>(&c,grid(out.len),block,out.arg());
            weighted_backward::launch::<$storage,$acc,CudaRuntime>(&c,grid(a.len),block,a.arg(),out.arg(),sc[0] as $acc,sc[1] as $acc,sc[2] as $acc,p[2]!=0,p[0] as u32,p[1]==3);
        } else {weighted_forward::launch::<$storage,$acc,CudaRuntime>(&c,grid(out.len),block,a.arg(),out.arg(),sc[0] as $acc,sc[1] as $acc,sc[2] as $acc,p[2]!=0,p[0] as u32,p[1]==3);}
    }};}
    if nearest {match a.dtype {0=>run_nearest!(f32,f32),1=>run_nearest!(f16,f32),2=>run_nearest!(bf16,f32),8=>run_nearest!(u8,i64),9=>run_nearest!(f64,f64),_=>panic!("nearest resizing requires floating or uint8 storage")}}
    else {match a.dtype {0=>run_weighted!(f32,f32),1=>run_weighted!(f16,f32),2=>run_weighted!(bf16,f32),9=>run_weighted!(f64,f64),_=>panic!("weighted resizing requires real floating storage")}}
    let launches=if antialias && !identity {if backward {4}else{3}}else{1};
    LAUNCHES.fetch_add(launches,Ordering::Relaxed);finish_dispatch(&c);
}
