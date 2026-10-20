// ATen CUDA margin-loss formulas/reductions; retained PyTorch license: bessel.rs.
use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn stored<S:Float,F:Float>(value:F)->F {F::cast_from(S::cast_from(value))}

#[ruda]
fn warp_sum<F:Float+RudaElement>(input:F)->F {
    let mut value=input;let mut shift=16u32;
    while shift>0 {value+=plane_shuffle_down(value,shift);shift/=2;}
    value
}

#[ruda]
fn block_sum<F:Float+RudaElement>(input:F)->F {
    let lane=UNIT_POS as usize;let mut shared=SharedMemory::<F>::new(4usize);
    let mut value=warp_sum::<F>(input);sync_ruda();
    if lane%32==0 {shared[lane/32]=value;}
    sync_ruda();value=F::new(0.0);
    if lane<4 {value=shared[lane];}
    if lane<32 {value=warp_sum::<F>(value);}
    value
}

#[ruda(launch)]
fn multi<S:Float+RudaElement,F:Float+RudaElement>(input:&Tensor<S>,target:&Tensor<i64>,weight:&Tensor<S>,grad:&Tensor<S>,out:&mut Tensor<S>,margin:F,#[comptime] p:u32,#[comptime] reduction:u32,#[comptime] weighted:bool,#[comptime] backward:bool) {
    let row=(RUDA_POS_X+RUDA_POS_Y*RUDA_COUNT_X) as usize;let lane=UNIT_POS as usize;
    if row<input.shape(0) {
        let width=input.shape(1);let base=row*width;let label=target[address(target,row)];
        if label<0 || label>=i64::cast_from(width) {
            intrinsic!(|scope| {scope.register(ruda_core::ir::Branch::Unreachable);});
        }
        let selected=F::cast_from(input[address(input,base+usize::cast_from(label))]);
        let boundary=stored::<S,F>(margin);let mut scale=F::new(1.0);
        if comptime!(weighted) {scale=F::cast_from(weight[address(weight,usize::cast_from(label))]);}
        let mut denominator=F::cast_from(width);
        if comptime!(reduction==1) {denominator=F::cast_from(input.len());}
        let normalizer=F::new(1.0)/denominator;
        let mut subtotal=F::new(0.0);let mut position=i64::cast_from(lane);
        let mut shared=SharedMemory::<F>::new(128usize);
        while position<i64::cast_from(width) {
            if position!=label {
                let at=base+usize::cast_from(position);let x=F::cast_from(input[address(input,at)]);
                let z=stored::<S,F>(stored::<S,F>(boundary-selected)+x);let mut value=F::new(0.0);
                if z>F::new(0.0) {
                    if comptime!(backward) {
                        value=normalizer;
                        if comptime!(p==2) {value=F::new(2.0)*normalizer*z;}
                        if comptime!(weighted) {value*=scale;}
                        value=stored::<S,F>(value);subtotal-=value;
                    } else {
                        value=z;
                        if comptime!(p==2) {value=stored::<S,F>(z*z);}
                        if comptime!(weighted) {value=stored::<S,F>(value*scale);}
                        subtotal+=value;
                    }
                }
                if comptime!(backward) {out[address(out,at)]=S::cast_from(value);}
            }
            position+=128;
        }
        shared[lane]=subtotal;sync_ruda();
        if lane==0 {
            let mut total=F::new(0.0);
            for thread in 0..128usize {total+=shared[thread];}
            if comptime!(backward) {out[address(out,base+usize::cast_from(label))]=S::cast_from(total);}
            else {out[address(out,row)]=S::cast_from(total/denominator);}
        }
        if comptime!(backward) {
            sync_ruda();let mut gradient=F::cast_from(grad[0]);
            if comptime!(reduction==0) {gradient=F::cast_from(grad[address(grad,row)]);}
            position=i64::cast_from(lane);
            while position<i64::cast_from(width) {
                let at=address(out,base+usize::cast_from(position));out[at]=S::cast_from(F::cast_from(out[at])*gradient);position+=128;
            }
        }
    }
}

#[ruda(launch)]
fn multilabel<S:Float+RudaElement,F:Float+RudaElement>(input:&Tensor<S>,target:&Tensor<i64>,mask:&mut Tensor<S>,grad:&Tensor<S>,out:&mut Tensor<S>,#[comptime] reduction:u32,#[comptime] backward:bool) {
    let row=(RUDA_POS_X+RUDA_POS_Y*RUDA_COUNT_X) as usize;let lane=UNIT_POS as usize;
    if row<input.shape(0) {
        let width=input.shape(1);let base=row*width;let mut position=i64::cast_from(lane);
        while position<i64::cast_from(width) {
            if comptime!(backward) {out[address(out,base+usize::cast_from(position))]=S::new(0.0);}
            else {mask[address(mask,base+usize::cast_from(position))]=S::new(0.0);}
            position+=128;
        }
        sync_ruda();
        if comptime!(!backward) {
            if lane==0 {
                let mut slot=0usize;
                while slot<width {
                    let label=target[address(target,base+slot)];
                    if label<0 {break;}
                    if label>=i64::cast_from(width) {
                        intrinsic!(|scope| {scope.register(ruda_core::ir::Branch::Unreachable);});
                    }
                    mask[address(mask,base+usize::cast_from(label))]=S::new(1.0);slot+=1;
                }
            }
            sync_ruda();
        }
        let mut denominator=F::cast_from(width);
        if comptime!(reduction==1) {denominator=F::cast_from(input.len());}
        let gain=stored::<S,F>(F::cast_from(1.0f64/f64::cast_from(denominator)));
        let mut subtotal=F::new(0.0);let mut slot=0usize;
        while slot<width {
            let label=target[address(target,base+slot)];
            if label<0 {break;}
            if label>=i64::cast_from(width) {
                intrinsic!(|scope| {scope.register(ruda_core::ir::Branch::Unreachable);});
            }
            let selected=F::cast_from(input[address(input,base+usize::cast_from(label))]);
            if comptime!(backward) {subtotal=F::new(0.0);}
            position=i64::cast_from(lane);
            while position<i64::cast_from(width) {
                let at=base+usize::cast_from(position);
                if i32::cast_from(mask[address(mask,at)])==0 {
                    let x=F::cast_from(input[address(input,at)]);let z=stored::<S,F>(stored::<S,F>(F::new(1.0)-selected)+x);
                    if z>F::new(0.0) {
                        if comptime!(backward) {
                            subtotal-=gain;
                            let output_at=address(out,at);out[output_at]=S::cast_from(F::cast_from(out[output_at])+gain);
                        } else {subtotal+=z;}
                    }
                }
                position+=128;
            }
            if comptime!(backward) {
                sync_ruda();let total=block_sum::<F>(subtotal);
                if lane==0 {
                    let at=address(out,base+usize::cast_from(label));out[at]=S::cast_from(F::cast_from(out[at])+stored::<S,F>(total));
                }
                sync_ruda();
            }
            slot+=1;
        }
        if comptime!(backward) {
            let mut gradient=F::cast_from(grad[0]);
            if comptime!(reduction==0) {gradient=F::cast_from(grad[address(grad,row)]);}
            position=i64::cast_from(lane);
            while position<i64::cast_from(width) {
                let at=address(out,base+usize::cast_from(position));out[at]=S::cast_from(F::cast_from(out[at])*gradient);position+=128;
            }
        } else {
            let total=block_sum::<F>(subtotal);
            if lane==0 {
                let mut value=total/F::cast_from(width);
                if comptime!(reduction==1) {value/=F::cast_from(input.shape(0));}
                out[address(out,row)]=S::cast_from(value);
            }
        }
    }
}

pub(super) fn dispatch(op:u32,xs:&[View],ints:&[i64],values:&[f64]) {
    assert!((361..=364).contains(&op));let backward=op==362 || op==364;let single=op<=362;
    assert_eq!(xs.len(),if backward {5}else{4});let input=&xs[0];let target=&xs[1];let out=xs.last().unwrap();
    assert_eq!(input.shape.len(),2);assert!(input.shape[1]>0);assert!(matches!(input.dtype,0|1|2|9));assert_eq!(target.dtype,4);
    let mut p=1u32;let reduction;let mut weighted=false;let mut margin=1.0f64;
    if single {
        assert_eq!(ints.len(),3);assert_eq!(values.len(),1);assert!(matches!(ints[0],1|2));assert!(matches!(ints[2],0|1));
        p=ints[0] as u32;reduction=ints[1];weighted=ints[2]!=0;margin=values[0];
        assert_eq!(target.shape,vec![input.shape[0]]);assert_eq!(xs[2].shape.len(),1);
        if weighted {assert_eq!(xs[2].len,input.shape[1]);}
    } else {
        assert_eq!(ints.len(),1);assert!(values.is_empty());reduction=ints[0];
        assert_eq!(target.shape,input.shape);assert_eq!(xs[2].shape,input.shape);
    }
    assert!((0..=2).contains(&reduction));for value in &xs[2..] {assert_eq!(value.dtype,input.dtype);}
    if backward {assert_eq!(out.shape,input.shape);assert_eq!(xs[3].shape,vec![if reduction==0 {input.shape[0]}else{1}]);}
    else {assert_eq!(out.shape,vec![input.shape[0]]);}
    if input.len==0 {return;}
    let count=u32::try_from(input.shape[0]).expect("margin loss batch count overflow");let x=count.min(1<<20);
    let grid=RudaCount::Static(x,count.div_ceil(x),1);let block=RudaDim::new_1d(128);let c=client();
    macro_rules! run {($s:ty,$f:ty)=>{unsafe {
        if single {multi::launch::<$s,$f,CudaRuntime>(&c,grid,block,input.arg(),target.arg(),xs[2].arg(),xs[3].arg(),out.arg(),margin as $f,p,reduction as u32,weighted,backward);}
        else {multilabel::launch::<$s,$f,CudaRuntime>(&c,grid,block,input.arg(),target.arg(),xs[2].arg(),xs[3].arg(),out.arg(),reduction as u32,backward);}
    }};}
    match input.dtype {0=>run!(f32,f32),1=>run!(f16,f32),2=>run!(bf16,f32),9=>run!(f64,f64),_=>unreachable!()}
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);
}
