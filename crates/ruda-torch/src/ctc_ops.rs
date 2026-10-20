// ATen CTC forward/backward recurrence; retained PyTorch license: bessel.rs.
use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn label<I:Numeric+RudaElement>(targets:&Tensor<I>,offset:i64,state:i64,blank:i64,classes:i64)->usize {
    let mut value=blank;
    if state%2!=0 {value=i64::cast_from(targets[address(targets,usize::cast_from(offset+state/2))]);}
    if value<0 || value>=classes {
        intrinsic!(|scope| {scope.register(ruda_core::ir::Branch::Unreachable);});
    }
    usize::cast_from(value)
}

#[ruda]
fn log_three<F:Float+RudaElement>(a:F,b:F,c:F)->F {
    let mut maximum=a;
    if b>maximum {maximum=b;}if c>maximum {maximum=c;}
    if maximum==F::new(f32::NEG_INFINITY) {maximum=F::new(0.0);}
    ((a-maximum).exp()+(b-maximum).exp()+(c-maximum).exp()).ln()+maximum
}

#[ruda]
fn log_two<F:Float+RudaElement>(a:F,b:F)->F {
    let mut maximum=b;if a>b {maximum=a;}
    if maximum==F::new(f32::NEG_INFINITY) {maximum=F::new(0.0);}
    ((a-maximum).exp()+(b-maximum).exp()).ln()+maximum
}

#[ruda(launch)]
fn forward<F:Float+RudaElement,I:Numeric+RudaElement>(probs:&Tensor<F>,targets:&Tensor<I>,metadata:&Tensor<i64>,nll:&mut Tensor<F>,alpha:&mut Tensor<F>,blank:i64) {
    let batch=(RUDA_POS_X+RUDA_POS_Y*RUDA_COUNT_X) as usize;let lane=UNIT_POS as usize;
    if batch<probs.shape(1) {
        let times=probs.shape(0);let classes=probs.shape(2);let width=alpha.shape(2);
        let input_length=metadata[address(metadata,batch*3)];let target_length=metadata[address(metadata,batch*3+1)];
        let target_offset=metadata[address(metadata,batch*3+2)];let end=target_length*2+1;
        let neginf=F::new(f32::NEG_INFINITY);let alpha_base=batch*times*width;
        for time in 0..times {
            let mut state=i64::cast_from(lane);
            while state<i64::cast_from(width) {
                let mut value=neginf;
                if i64::cast_from(time)<input_length && state<end {
                    let current=label::<I>(targets,target_offset,state,blank,i64::cast_from(classes));
                    let probability=probs[time*probs.stride(0)+batch*probs.stride(1)+current*probs.stride(2)];
                    if time==0 {if state<2 {value=probability;}}
                    else {
                        let previous=alpha_base+(time-1)*width+usize::cast_from(state);
                        let one=alpha[address(alpha,previous)];let mut two=neginf;let mut three=neginf;
                        if state>0 {two=alpha[address(alpha,previous-1)];}
                        if state>1 {
                            if label::<I>(targets,target_offset,state-2,blank,i64::cast_from(classes))!=current {three=alpha[address(alpha,previous-2)];}
                        }
                        value=log_three::<F>(one,two,three)+probability;
                    }
                }
                alpha[address(alpha,alpha_base+time*width+usize::cast_from(state))]=value;
                state+=256;
            }
            sync_ruda();
        }
        if lane==0 {
            let mut likelihood=neginf;
            if input_length==0 {if target_length==0 {likelihood=F::new(0.0);}}
            else {
                let last=alpha_base+usize::cast_from(input_length-1)*width+usize::cast_from(end-1);
                let one=alpha[address(alpha,last)];let mut two=neginf;
                if target_length>0 {two=alpha[address(alpha,last-1)];}
                likelihood=log_two::<F>(one,two);
            }
            nll[address(nll,batch)]=-likelihood;
        }
    }
}

#[ruda(launch)]
fn backward_beta<F:Float+RudaElement,I:Numeric+RudaElement>(probs:&Tensor<F>,targets:&Tensor<I>,metadata:&Tensor<i64>,beta:&mut Tensor<F>,blank:i64) {
    let batch=(RUDA_POS_X+RUDA_POS_Y*RUDA_COUNT_X) as usize;let lane=UNIT_POS as usize;
    if batch<probs.shape(1) {
        let times=probs.shape(0);let classes=probs.shape(2);let width=beta.shape(2);
        let input_length=metadata[address(metadata,batch*3)];let target_length=metadata[address(metadata,batch*3+1)];
        let target_offset=metadata[address(metadata,batch*3+2)];let end=target_length*2+1;
        let neginf=F::new(f32::NEG_INFINITY);let beta_base=batch*times*width;let mut time=i64::cast_from(times)-1;
        while time>=0 {
            let mut state=i64::cast_from(lane);
            while state<i64::cast_from(width) {
                let mut value=neginf;
                if time<input_length && state<end {
                    let current=label::<I>(targets,target_offset,state,blank,i64::cast_from(classes));
                    let probability=probs[usize::cast_from(time)*probs.stride(0)+batch*probs.stride(1)+current*probs.stride(2)];
                    if time==input_length-1 {if state>=end-2 {value=probability;}}
                    else {
                        let next=beta_base+usize::cast_from(time+1)*width+usize::cast_from(state);
                        let one=beta[address(beta,next)];let mut two=neginf;let mut three=neginf;
                        if state+1<end {two=beta[address(beta,next+1)];}
                        if state+2<end {
                            if label::<I>(targets,target_offset,state+2,blank,i64::cast_from(classes))!=current {three=beta[address(beta,next+2)];}
                        }
                        value=log_three::<F>(one,two,three)+probability;
                    }
                }
                beta[address(beta,beta_base+usize::cast_from(time)*width+usize::cast_from(state))]=value;
                state+=256;
            }
            sync_ruda();time-=1;
        }
    }
}

#[ruda(launch)]
fn collect<F:Float+RudaElement,I:Numeric+RudaElement>(probs:&Tensor<F>,targets:&Tensor<I>,metadata:&Tensor<i64>,nll:&Tensor<F>,alpha:&Tensor<F>,beta:&Tensor<F>,grad:&Tensor<F>,out:&mut Tensor<F>,blank:i64,#[comptime] zero_infinity:bool) {
    let index=ABSOLUTE_POS as usize;let batches=probs.shape(1);let times=probs.shape(0);let classes=probs.shape(2);
    if index<times*batches {
        let batch=index%batches;let time=index/batches;let input_length=metadata[address(metadata,batch*3)];
        let likelihood=nll[address(nll,batch)];let mut active=i64::cast_from(time)<input_length;
        if comptime!(zero_infinity) {if likelihood==F::new(f32::INFINITY) {active=false;}}
        let output_base=time*out.stride(0)+batch*out.stride(1);let neginf=F::new(f32::NEG_INFINITY);
        if active {
            for character in 0..classes {out[output_base+character*out.stride(2)]=neginf;}
            let target_length=metadata[address(metadata,batch*3+1)];let target_offset=metadata[address(metadata,batch*3+2)];
            let base=(batch*times+time)*alpha.shape(2);let mut state=0i64;
            while state<target_length*2+1 {
                let character=label::<I>(targets,target_offset,state,blank,i64::cast_from(classes));
                let at=output_base+character*out.stride(2);let old=out[at];
                let value=alpha[address(alpha,base+usize::cast_from(state))]+beta[address(beta,base+usize::cast_from(state))];
                if old==neginf {out[at]=value;}else{out[at]=log_two::<F>(old,value);}
                state+=1;
            }
            let g=grad[address(grad,batch)];let probability_base=time*probs.stride(0)+batch*probs.stride(1);
            for character in 0..classes {
                let at=output_base+character*out.stride(2);let lp=probs[probability_base+character*probs.stride(2)];
                out[at]=(lp.exp()-(out[at]+likelihood-lp).exp())*g;
            }
        } else {
            for character in 0..classes {out[output_base+character*out.stride(2)]=F::new(0.0);}
        }
    }
}

#[ruda(launch)]
fn collect_nonblank<F:Float+RudaElement,I:Numeric+RudaElement>(probs:&Tensor<F>,targets:&Tensor<I>,metadata:&Tensor<i64>,nll:&Tensor<F>,alpha:&Tensor<F>,beta:&Tensor<F>,grad:&Tensor<F>,out:&mut Tensor<Atomic<F>>,blank:i64,#[comptime] zero_infinity:bool) {
    let index=ABSOLUTE_POS as usize;let maximum=alpha.shape(2)/2;
    if index<probs.shape(1)*maximum {
        let batch=index/maximum;let state=index%maximum;let length=metadata[address(metadata,batch*3+1)];
        let likelihood=nll[address(nll,batch)];let mut active=i64::cast_from(state)<length;
        if comptime!(zero_infinity) {if likelihood==F::new(f32::INFINITY) {active=false;}}
        if active {
            let offset=metadata[address(metadata,batch*3+2)];let character=label::<I>(targets,offset,i64::cast_from(state)*2+1,blank,i64::cast_from(probs.shape(2)));
            let input_length=metadata[address(metadata,batch*3)];let g=grad[address(grad,batch)];let mut time=0i64;
            while time<input_length {
                let position=(batch*probs.shape(0)+usize::cast_from(time))*alpha.shape(2)+state*2+1;
                let lp=probs[usize::cast_from(time)*probs.stride(0)+batch*probs.stride(1)+character*probs.stride(2)];
                let value=-(alpha[address(alpha,position)]+beta[address(beta,position)]+likelihood-lp).exp()*g;
                out[usize::cast_from(time)*out.stride(0)+batch*out.stride(1)+character*out.stride(2)].fetch_add(value);
                time+=1;
            }
        }
    }
}

#[ruda(launch)]
fn clear_padding<F:Float+RudaElement>(out:&mut Tensor<F>,metadata:&Tensor<i64>) {
    let index=ABSOLUTE_POS as usize;
    if index<out.len() {
        let batch=index/out.shape(2)%out.shape(1);let time=index/(out.shape(2)*out.shape(1));
        if i64::cast_from(time)>=metadata[address(metadata,batch*3)] {out[address(out,index)]=F::new(0.0);}
    }
}

fn grid(size:usize)->RudaCount {RudaCount::Static(u32::try_from(size.div_ceil(256)).expect("CTC launch overflow"),1,1)}

pub(super) fn dispatch(op:u32,xs:&[View],ints:&[i64]) {
    assert!((356..=360).contains(&op));let c=client();let block=RudaDim::new_1d(256);
    if op==360 {
        assert!(ints.is_empty());assert_eq!(xs.len(),2);assert_eq!(xs[0].shape.len(),3);assert_eq!(xs[1].dtype,4);
        assert_eq!(xs[1].shape,vec![xs[0].shape[1],3]);assert!(matches!(xs[0].dtype,0|9));
        if xs[0].len==0 {return;}
        unsafe {if xs[0].dtype==0 {clear_padding::launch::<f32,CudaRuntime>(&c,grid(xs[0].len),block,xs[0].arg(),xs[1].arg());}
        else {clear_padding::launch::<f64,CudaRuntime>(&c,grid(xs[0].len),block,xs[0].arg(),xs[1].arg());}}
        LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);return;
    }
    assert_eq!(xs.len(),match op {356=>5,357=>4,_=>8});assert_eq!(ints.len(),if op<=357 {1}else{2});
    let probs=&xs[0];let targets=&xs[1];let metadata=&xs[2];let scalar=probs.dtype;
    assert!(matches!(scalar,0|9));assert_eq!(probs.shape.len(),3);assert!(probs.len>0);
    assert_eq!(targets.shape.len(),1);assert!(matches!(targets.dtype,4|5));assert_eq!(metadata.dtype,4);
    let times=probs.shape[0];let batches=probs.shape[1];let classes=probs.shape[2];
    assert_eq!(metadata.shape,vec![batches,3]);assert!(ints[0]>=0 && (ints[0] as usize)<classes);let blank=ints[0];
    let mut zero_infinity=false;if op>=358 {assert!(matches!(ints[1],0|1));zero_infinity=ints[1]!=0;}
    for value in &xs[3..] {assert_eq!(value.dtype,scalar);}
    let states=if op==357 {&xs[3]}else{&xs[4]};assert_eq!(states.shape.len(),3);
    assert_eq!(states.shape[0],batches);assert_eq!(states.shape[1],times);assert!(states.shape[2]>0 && states.shape[2]%2==1);
    if op!=357 {assert_eq!(xs[3].shape,vec![batches]);}
    if op>=358 {assert_eq!(xs[5].shape,states.shape);assert_eq!(xs[6].shape,vec![batches]);assert_eq!(xs[7].shape,probs.shape);}
    let rows=u32::try_from(batches).expect("CTC batch count overflow");let x=rows.min(1<<20);
    let batch_grid=RudaCount::Static(x,rows.div_ceil(x),1);
    let count=if op==358 {times*batches}else{batches*(states.shape[2]/2)};
    if op==359 && count==0 {return;}
    macro_rules! run {($f:ty,$i:ty)=>{unsafe {
        match op {
            356=>forward::launch::<$f,$i,CudaRuntime>(&c,batch_grid,block,probs.arg(),targets.arg(),metadata.arg(),xs[3].arg(),xs[4].arg(),blank),
            357=>backward_beta::launch::<$f,$i,CudaRuntime>(&c,batch_grid,block,probs.arg(),targets.arg(),metadata.arg(),xs[3].arg(),blank),
            358=>collect::launch::<$f,$i,CudaRuntime>(&c,grid(count),block,probs.arg(),targets.arg(),metadata.arg(),xs[3].arg(),xs[4].arg(),xs[5].arg(),xs[6].arg(),xs[7].arg(),blank,zero_infinity),
            359=>collect_nonblank::launch::<$f,$i,CudaRuntime>(&c,grid(count),block,probs.arg(),targets.arg(),metadata.arg(),xs[3].arg(),xs[4].arg(),xs[5].arg(),xs[6].arg(),xs[7].arg(),blank,zero_infinity),
            _=>unreachable!(),
        };
    }};}
    match (scalar,targets.dtype) {(0,4)=>run!(f32,i64),(0,5)=>run!(f32,i32),(9,4)=>run!(f64,i64),(9,5)=>run!(f64,i32),_=>unreachable!()}
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);
}
