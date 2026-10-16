use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn unnormalize<E:Float+RudaElement,F:Float+RudaElement>(value:E,size:usize,#[comptime] align:bool)->(E,E) {
    let mut coordinate=F::cast_from(value)+F::new(1.0);let mut multiplier=E::new(0.0);
    if comptime!(align) {
        coordinate=coordinate/F::new(2.0)*F::cast_from(size-1);
        multiplier=E::cast_from(size-1)/E::new(2.0);
    } else {
        coordinate=(coordinate*F::cast_from(size)-F::new(1.0))/F::new(2.0);
        multiplier=E::cast_from(size)/E::new(2.0);
    }
    (E::cast_from(coordinate),multiplier)
}

#[ruda]
fn padded<E:Float+RudaElement,F:Float+RudaElement>(value:E,size:usize,#[comptime] padding:u32,#[comptime] align:bool,#[comptime] backward:bool)->(E,E) {
    let mut coordinate=value;let mut multiplier=E::new(1.0);
    if comptime!(padding==2) {
        let mut low=E::new(-0.5);let mut span=E::cast_from(2*size)/E::new(2.0);
        if comptime!(align) {low=E::new(0.0);span=E::cast_from(2*(size-1))/E::new(2.0);}
        if comptime!(align) && size==1 {coordinate=E::new(0.0);multiplier=E::new(0.0);}
        else {
            coordinate-=low;
            if coordinate<E::new(0.0) {coordinate=-coordinate;multiplier=E::new(-1.0);}
            let extra=coordinate%span;let flips=i32::cast_from((coordinate/span).floor());
            if flips%2==0 {coordinate=extra+low;}
            else {coordinate=span-extra+low;multiplier=-multiplier;}
        }
    }
    if comptime!(padding!=0) {
        let limit=E::cast_from(size-1);
        if comptime!(backward) {
            if coordinate<=E::new(0.0) {coordinate=E::new(0.0);multiplier*=E::new(0.0);}
            else if coordinate>=limit {coordinate=limit;multiplier*=E::new(0.0);}
        } else {coordinate=E::min(limit,E::max(coordinate,E::new(0.0)));}
    }
    let wide=F::cast_from(coordinate);
    if wide!=wide || wide>F::cast_from(2147483646.0f64) || wide<F::new(-2147483648.0) {coordinate=E::new(-100.0);}
    (coordinate,multiplier)
}

#[ruda]
fn source<E:Float+RudaElement,F:Float+RudaElement>(value:E,size:usize,#[comptime] padding:u32,#[comptime] align:bool,#[comptime] backward:bool,#[comptime] cubic:bool)->(E,E) {
    let (raw,scale)=unnormalize::<E,F>(value,size,align);let mut coordinate=raw;let mut multiplier=scale;
    if comptime!(!cubic) {
        let (mapped,derivative)=padded::<E,F>(raw,size,padding,align,backward);
        coordinate=mapped;multiplier*=derivative;
    }
    (coordinate,multiplier)
}

#[ruda]
fn inside<S:Float+RudaElement>(a:&Tensor<S>,z:i64,y:i64,x:i64)->bool {
    x>=0 && x<i64::cast_from(a.shape(4)) && y>=0 && y<i64::cast_from(a.shape(3)) && z>=0 && z<i64::cast_from(a.shape(2))
}

#[ruda]
fn load<S:Float+RudaElement>(a:&Tensor<S>,plane:usize,z:i64,y:i64,x:i64)->S {
    let at=((plane*a.shape(2)+usize::cast_from(z))*a.shape(3)+usize::cast_from(y))*a.shape(4)+usize::cast_from(x);
    a[address(a,at)]
}

#[ruda]
fn add<S:Float+RudaElement>(out:&mut Tensor<Atomic<S>>,plane:usize,z:i64,y:i64,x:i64,value:S) {
    let channels=out.shape(1);
    let at=plane/channels*out.stride(0)+plane%channels*out.stride(1)
        +usize::cast_from(z)*out.stride(2)+usize::cast_from(y)*out.stride(3)+usize::cast_from(x)*out.stride(4);
    out[at].fetch_add(value);
}

#[ruda]
fn bounded<S:Float+RudaElement,F:Float+RudaElement>(a:&Tensor<S>,plane:usize,y:S,x:S,#[comptime] padding:u32,#[comptime] align:bool)->S {
    let (px,_mx)=padded::<S,F>(x,a.shape(4),padding,align,false);
    let (py,_my)=padded::<S,F>(y,a.shape(3),padding,align,false);
    let ix=i64::cast_from(px);let iy=i64::cast_from(py);let mut value=S::new(0.0);
    if inside::<S>(a,0,iy,ix) {value=load::<S>(a,plane,0,iy,ix);}
    value
}

#[ruda]
fn cubic_coefficient<E:Float+RudaElement>(t:E,index:usize)->E {
    let a=E::new(-0.75);let mut x=t;let mut result=E::new(0.0);
    if index>=2 {x=E::new(1.0)-t;}
    if index==0 || index==3 {
        x+=E::new(1.0);result=((a*x-E::new(5.0)*a)*x+E::new(8.0)*a)*x-E::new(4.0)*a;
    } else {result=((a+E::new(2.0))*x-(a+E::new(3.0)))*x*x+E::new(1.0);}
    result
}

#[ruda]
fn cubic_derivative<E:Float+RudaElement>(t:E,index:usize)->E {
    let a=E::new(-0.75);let mut result=E::new(0.0);
    if index==0 {let x=E::new(-1.0)-t;result=(E::new(-3.0)*a*x-E::new(10.0)*a)*x-E::new(8.0)*a;}
    else if index==1 {let x=-t;result=(E::new(-3.0)*(a+E::new(2.0))*x-E::new(2.0)*(a+E::new(3.0)))*x;}
    else if index==2 {let x=E::new(1.0)-t;result=(E::new(3.0)*(a+E::new(2.0))*x-E::new(2.0)*(a+E::new(3.0)))*x;}
    else {let x=E::new(2.0)-t;result=(E::new(3.0)*a*x-E::new(10.0)*a)*x+E::new(8.0)*a;}
    result
}

#[ruda(launch)]
fn forward<S:Float+RudaElement,F:Float+RudaElement>(a:&Tensor<S>,grid:&Tensor<S>,out:&mut Tensor<S>,#[comptime] rank:u32,#[comptime] mode:u32,#[comptime] padding:u32,#[comptime] align:bool) {
    let p=ABSOLUTE_POS as usize;let dimensions=comptime!(rank as usize);
    if p<grid.len()/dimensions {
        let spatial=grid.shape(1)*grid.shape(2)*grid.shape(3);let n=p/spatial;let location=p%spatial;
        let (ix,_mx)=source::<F,F>(F::cast_from(grid[address(grid,p*dimensions)]),a.shape(4),padding,align,false,mode==2);
        let (iy,_my)=source::<F,F>(F::cast_from(grid[address(grid,p*dimensions+1)]),a.shape(3),padding,align,false,mode==2);
        let mut iz=F::new(0.0);
        if comptime!(rank==3) {
            let (z,_mz)=source::<F,F>(F::cast_from(grid[address(grid,p*dimensions+2)]),a.shape(2),padding,align,false,false);iz=z;
        }
        if comptime!(mode==2) {
            let xbase=ix.floor();let ybase=iy.floor();let tx=ix-xbase;let ty=iy-ybase;
            let c0=cubic_coefficient::<F>(tx,0);let c1=cubic_coefficient::<F>(tx,1);let c2=cubic_coefficient::<F>(tx,2);let c3=cubic_coefficient::<F>(tx,3);
            for channel in 0..a.shape(1) {
                let plane=n*a.shape(1)+channel;let mut rows=Array::<F>::new(4);
                for k in 0..4usize {
                    let y=S::cast_from(ybase-F::new(1.0)+F::cast_from(k));
                    let v0=F::cast_from(bounded::<S,F>(a,plane,y,S::cast_from(xbase-F::new(1.0)),padding,align));
                    let v1=F::cast_from(bounded::<S,F>(a,plane,y,S::cast_from(xbase+F::new(0.0)),padding,align));
                    let v2=F::cast_from(bounded::<S,F>(a,plane,y,S::cast_from(xbase+F::new(1.0)),padding,align));
                    let v3=F::cast_from(bounded::<S,F>(a,plane,y,S::cast_from(xbase+F::new(2.0)),padding,align));
                    rows[k]=v0*c0+v1*c1+v2*c2+v3*c3;
                }
                let value=rows[0]*cubic_coefficient::<F>(ty,0)+rows[1]*cubic_coefficient::<F>(ty,1)+rows[2]*cubic_coefficient::<F>(ty,2)+rows[3]*cubic_coefficient::<F>(ty,3);
                out[address(out,plane*spatial+location)]=S::cast_from(value);
            }
        } else if comptime!(mode==1) {
            let x=i64::cast_from(ix.round());let y=i64::cast_from(iy.round());let z=i64::cast_from(iz.round());
            for channel in 0..a.shape(1) {
                let plane=n*a.shape(1)+channel;let mut value=S::new(0.0);
                if inside::<S>(a,z,y,x) {value=load::<S>(a,plane,z,y,x);}
                out[address(out,plane*spatial+location)]=value;
            }
        } else {
            let xbase=i64::cast_from(ix.floor());let ybase=i64::cast_from(iy.floor());let zbase=i64::cast_from(iz.floor());
            let wx0=F::cast_from(xbase+1)-ix;let wx1=ix-F::cast_from(xbase);
            let wy0=F::cast_from(ybase+1)-iy;let wy1=iy-F::cast_from(ybase);
            let wz0=F::cast_from(zbase+1)-iz;let wz1=iz-F::cast_from(zbase);
            let depths=comptime!(if rank==3 {2usize}else{1usize});
            for channel in 0..a.shape(1) {
                let plane=n*a.shape(1)+channel;let mut value=F::new(0.0);
                for kz in 0..depths {for ky in 0..2usize {for kx in 0..2usize {
                    let z=zbase+i64::cast_from(kz);let y=ybase+i64::cast_from(ky);let x=xbase+i64::cast_from(kx);
                    if inside::<S>(a,z,y,x) {
                        let mut wx=wx0;let mut wy=wy0;let mut wz=wz0;
                        if kx==1 {wx=wx1;}if ky==1 {wy=wy1;}if kz==1 {wz=wz1;}
                        let mut weight=wx*wy;if comptime!(rank==3) {weight*=wz;}
                        value+=F::cast_from(load::<S>(a,plane,z,y,x))*weight;
                    }
                }}}
                out[address(out,plane*spatial+location)]=S::cast_from(value);
            }
        }
    }
}

#[ruda(launch)]
fn clear<S:Float+RudaElement>(out:&mut Tensor<S>) {
    let p=ABSOLUTE_POS as usize;if p<out.len() {out[address(out,p)]=S::new(0.0);}
}

#[ruda(launch)]
fn backward<S:Float+RudaElement,F:Float+RudaElement>(a:&Tensor<S>,grid:&Tensor<S>,grad:&Tensor<S>,dx:&mut Tensor<Atomic<S>>,dgrid:&mut Tensor<S>,#[comptime] rank:u32,#[comptime] mode:u32,#[comptime] padding:u32,#[comptime] align:bool,#[comptime] want_dx:bool) {
    let p=ABSOLUTE_POS as usize;let dimensions=comptime!(rank as usize);
    if p<grid.len()/dimensions {
        let spatial=grid.shape(1)*grid.shape(2)*grid.shape(3);let n=p/spatial;let location=p%spatial;
        let (ix,mx)=source::<S,F>(grid[address(grid,p*dimensions)],a.shape(4),padding,align,true,mode==2);
        let (iy,my)=source::<S,F>(grid[address(grid,p*dimensions+1)],a.shape(3),padding,align,true,mode==2);
        let mut iz=S::new(0.0);let mut mz=S::new(0.0);
        if comptime!(rank==3) {
            let (z,m)=source::<S,F>(grid[address(grid,p*dimensions+2)],a.shape(2),padding,align,true,false);iz=z;mz=m;
        }
        let mut gx=S::new(0.0);let mut gy=S::new(0.0);let mut gz=S::new(0.0);
        if comptime!(mode==2) {
            let xbase=ix.floor();let ybase=iy.floor();let tx=ix-xbase;let ty=iy-ybase;
            let mut cx=Array::<S>::new(4);let mut cy=Array::<S>::new(4);
            let mut dcx=Array::<S>::new(4);let mut dcy=Array::<S>::new(4);
            for k in 0..4usize {cx[k]=cubic_coefficient::<S>(tx,k);cy[k]=cubic_coefficient::<S>(ty,k);dcx[k]=cubic_derivative::<S>(tx,k);dcy[k]=cubic_derivative::<S>(ty,k);}
            for channel in 0..a.shape(1) {
                let plane=n*a.shape(1)+channel;let go=grad[address(grad,plane*spatial+location)];
                for kx in 0..4usize {for ky in 0..4usize {
                    let x=xbase-S::new(1.0)+S::cast_from(kx);let y=ybase-S::new(1.0)+S::cast_from(ky);
                    let (px,_pmx)=padded::<S,F>(x,a.shape(4),padding,align,false);
                    let (py,_pmy)=padded::<S,F>(y,a.shape(3),padding,align,false);
                    let xx=i64::cast_from(px);let yy=i64::cast_from(py);let mut value=S::new(0.0);
                    if inside::<S>(a,0,yy,xx) {
                        value=load::<S>(a,plane,0,yy,xx);
                        if comptime!(want_dx) {add::<S>(dx,plane,0,yy,xx,go*cx[kx]*cy[ky]);}
                    }
                    gx-=value*dcx[kx]*cy[ky]*go;gy-=value*dcy[ky]*cx[kx]*go;
                }}
            }
        } else if comptime!(mode==1) {
            if comptime!(want_dx) {
                let x=i64::cast_from(ix.round());let y=i64::cast_from(iy.round());let z=i64::cast_from(iz.round());
                if inside::<S>(a,z,y,x) {for channel in 0..a.shape(1) {
                    let plane=n*a.shape(1)+channel;add::<S>(dx,plane,z,y,x,grad[address(grad,plane*spatial+location)]);
                }}
            }
        } else {
            let xbase=i64::cast_from(ix.floor());let ybase=i64::cast_from(iy.floor());let zbase=i64::cast_from(iz.floor());
            let wx0=S::cast_from(xbase+1)-ix;let wx1=ix-S::cast_from(xbase);
            let wy0=S::cast_from(ybase+1)-iy;let wy1=iy-S::cast_from(ybase);
            let wz0=S::cast_from(zbase+1)-iz;let wz1=iz-S::cast_from(zbase);
            let depths=comptime!(if rank==3 {2usize}else{1usize});
            for channel in 0..a.shape(1) {
                let plane=n*a.shape(1)+channel;let go=grad[address(grad,plane*spatial+location)];
                for kz in 0..depths {for ky in 0..2usize {for kx in 0..2usize {
                    let z=zbase+i64::cast_from(kz);let y=ybase+i64::cast_from(ky);let x=xbase+i64::cast_from(kx);
                    if inside::<S>(a,z,y,x) {
                        let mut wx=wx0;let mut wy=wy0;let mut wz=wz0;
                        if kx==1 {wx=wx1;}if ky==1 {wy=wy1;}if kz==1 {wz=wz1;}
                        if comptime!(want_dx) {
                            let mut weight=wx*wy;if comptime!(rank==3) {weight*=wz;}
                            add::<S>(dx,plane,z,y,x,weight*go);
                        }
                        let value=load::<S>(a,plane,z,y,x);let mut vx=value*wy;let mut vy=value*wx;
                        if comptime!(rank==3) {
                            vx*=wz;vy*=wz;let vz=value*wx*wy*go;
                            if kz==0 {gz-=vz;}else{gz+=vz;}
                        }
                        vx*=go;vy*=go;
                        if kx==0 {gx-=vx;}else{gx+=vx;}
                        if ky==0 {gy-=vy;}else{gy+=vy;}
                    }
                }}}
            }
        }
        if comptime!(mode!=1) {gx=mx*gx;gy=my*gy;if comptime!(rank==3) {gz=mz*gz;}}
        dgrid[address(dgrid,p*dimensions)]=gx;dgrid[address(dgrid,p*dimensions+1)]=gy;
        if comptime!(rank==3) {dgrid[address(dgrid,p*dimensions+2)]=gz;}
    }
}

fn launch_grid(n:usize)->RudaCount {RudaCount::Static(u32::try_from(n.div_ceil(256)).expect("grid sampling launch overflow"),1,1)}

pub(super) fn dispatch(op:u32,xs:&[View],p:&[i64]) {
    assert!(op==331 || op==332);let back=op==332;
    assert_eq!(xs.len(),if back {5}else{3});assert_eq!(p.len(),if back {5}else{4});
    assert!(p[0]==2 || p[0]==3);assert!((0..=2).contains(&p[1]));assert!(p[1]!=2 || p[0]==2);
    assert!((0..=2).contains(&p[2]));assert!(p[3]==0 || p[3]==1);
    let want_dx=back && p[4]!=0;if back {assert!(p[4]==0 || p[4]==1);}
    let a=&xs[0];let grid=&xs[1];let out=&xs[2];assert!(matches!(a.dtype,0|1|2|9));
    for view in xs {assert_eq!(view.dtype,a.dtype);}
    assert_eq!(a.shape.len(),5);assert_eq!(grid.shape.len(),5);assert_eq!(out.shape.len(),5);
    assert!(a.shape[2..].iter().all(|&n|n>0));assert_eq!(a.shape[0],grid.shape[0]);assert_eq!(grid.shape[4],p[0] as usize);
    assert_eq!(out.shape[..2],a.shape[..2]);assert_eq!(out.shape[2..],grid.shape[1..4]);
    if p[0]==2 {assert_eq!(a.shape[2],1);assert_eq!(grid.shape[1],1);}
    if back {assert_eq!(xs[4].shape,grid.shape);if want_dx {assert_eq!(xs[3].shape,a.shape);}else{assert_eq!(xs[3].len,0);}}
    let points=grid.len/p[0] as usize;let c=client().fixed_execution_queue();let block=RudaDim::new_1d(256);
    macro_rules! run {($s:ty,$f:ty)=>{unsafe {
        if back {
            if want_dx && xs[3].len!=0 {clear::launch::<$s,CudaRuntime>(&c,launch_grid(xs[3].len),block,xs[3].arg());}
            if points!=0 {backward::launch::<$s,$f,CudaRuntime>(&c,launch_grid(points),block,a.arg(),grid.arg(),out.arg(),xs[3].arg(),xs[4].arg(),p[0] as u32,p[1] as u32,p[2] as u32,p[3]!=0,want_dx);}
        } else if out.len!=0 {forward::launch::<$s,$f,CudaRuntime>(&c,launch_grid(points),block,a.arg(),grid.arg(),out.arg(),p[0] as u32,p[1] as u32,p[2] as u32,p[3]!=0);}
    }};}
    match a.dtype {0=>run!(f32,f32),1=>run!(f16,f32),2=>run!(bf16,f32),9=>run!(f64,f64),_=>unreachable!()}
    let launches=if back {u64::from(want_dx && xs[3].len!=0)+u64::from(points!=0)}else{u64::from(out.len!=0)};
    LAUNCHES.fetch_add(launches,Ordering::Relaxed);finish_dispatch(&c);
}
