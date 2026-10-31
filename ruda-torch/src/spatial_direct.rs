use super::*;

fn indexable(view:&View)->bool {
    view.len<=u32::MAX as usize && view.shape.iter().zip(&view.strides).try_fold(0usize,|offset,(&size,&stride)|
        offset.checked_add(size.saturating_sub(1).checked_mul(stride)?)).is_some_and(|offset|offset<=u32::MAX as usize)
}

#[ruda(launch)]
fn forward<F:Float+RudaElement>(input:&Tensor<F>,weight:&Tensor<F>,output:&mut Tensor<F>,
    spatial:usize,tiles:usize,channels_per_group:usize,
    #[comptime] stride:Vec<usize>,#[comptime] padding:Vec<usize>,#[comptime] dilation:Vec<usize>,
    #[comptime] chunks:usize)
{
    let tile=RUDA_POS_X as usize;
    let row=tile/tiles;let channel=row%output.shape(1);let batch=row/output.shape(1);
    let first=tile%tiles*(128usize*chunks)+UNIT_POS as usize;
    let input_channel=channel/channels_per_group*weight.shape(1);
    let mut sums=Array::<F>::new(chunks);
    let mut coordinates=Array::<i64>::new(chunks*stride.len());
    let mut destinations=Array::<usize>::new(chunks);
    #[unroll] for item in 0usize..chunks {
        sums[item]=F::cast_from(0);
        let mut pos=first+item*128usize;
        let mut destination=batch*output.stride(0)+channel*output.stride(1);
        #[unroll] for reverse in 0usize..stride.len() {
            let axis=comptime!(stride.len()-1-reverse);
            let coordinate=pos%output.shape(axis+2);pos/=output.shape(axis+2);
            coordinates[item*stride.len()+axis]=i64::cast_from(coordinate*comptime!(stride[axis]))-comptime!(padding[axis]) as i64;
            destination+=coordinate*output.stride(axis+2);
        }
        destinations[item]=destination;
    }
    let mut volume=1usize;
    #[unroll] for axis in 0usize..stride.len() {volume*=weight.shape(axis+2);}
    // Same spatial-major, input-channel-minor accumulation as conv_direct.
    for tap in 0usize..volume {
        let mut position=tap;let mut weight_at=channel*weight.stride(0);
        let mut delta=Array::<i64>::new(stride.len());
        #[unroll] for reverse in 0usize..stride.len() {
            let axis=comptime!(stride.len()-1-reverse);
            let coordinate=position%weight.shape(axis+2);position/=weight.shape(axis+2);
            delta[axis]=i64::cast_from(coordinate*comptime!(dilation[axis]));
            weight_at+=coordinate*weight.stride(axis+2);
        }
        let mut bases=Array::<usize>::new(chunks);let mut valid=Array::<bool>::new(chunks);
        #[unroll] for item in 0usize..chunks {
            let mut at=batch*input.stride(0)+input_channel*input.stride(1);
            let mut inside=first+item*128usize<spatial;
            #[unroll] for axis in 0usize..stride.len() {
                let coordinate=coordinates[item*stride.len()+axis]+delta[axis];
                inside=inside && coordinate>=0i64 && coordinate<i64::cast_from(input.shape(axis+2));
                if coordinate>=0i64 {at+=usize::cast_from(coordinate)*input.stride(axis+2);}
            }
            bases[item]=at;valid[item]=inside;
        }
        for ic in 0usize..weight.shape(1) {
            let w=weight[weight_at+ic*weight.stride(1)];
            #[unroll] for item in 0usize..chunks {
                if valid[item] {let product=input[bases[item]+ic*input.stride(1)]*w;sums[item]+=product;}
            }
        }
    }
    #[unroll] for item in 0usize..chunks {
        if first+item*128usize<spatial {output[destinations[item]]=sums[item];}
    }
}

pub(super) fn launch(a:&View,b:&View,out:&View,p:&[i64])->bool {
    let rank=a.shape.len();
    if !(3..=5).contains(&rank) || !out.contiguous() || ![a,b,out].iter().all(|v|indexable(v)) {return false;}
    let n=rank-2;
    assert_eq!(p.len(),4*n+1);assert_eq!(b.shape.len(),rank);assert_eq!(out.shape.len(),rank);
    assert_eq!(a.dtype,b.dtype);assert_eq!(a.dtype,out.dtype);assert_eq!(a.shape[0],out.shape[0]);
    let groups=usize::try_from(p[4*n]).expect("negative convolution groups");
    assert!(groups>0 && a.shape[1]%groups==0 && b.shape[0]%groups==0);
    assert_eq!(b.shape[1],a.shape[1]/groups);assert_eq!(out.shape[1],b.shape[0]);
    let stride:Vec<_>=p[..n].iter().map(|&v|usize::try_from(v).expect("negative stride")).collect();
    let padding:Vec<_>=p[n..2*n].iter().map(|&v|usize::try_from(v).expect("negative padding")).collect();
    let dilation:Vec<_>=p[2*n..3*n].iter().map(|&v|usize::try_from(v).expect("negative dilation")).collect();
    for axis in 0..n {
        assert!(stride[axis]>0 && dilation[axis]>0 && b.shape[axis+2]>0);
        let padded=a.shape[axis+2].checked_add(padding[axis].checked_mul(2).expect("padding overflow")).expect("convolution extent overflow");
        let filter=dilation[axis].checked_mul(b.shape[axis+2]-1).and_then(|v|v.checked_add(1)).expect("filter extent overflow");
        assert!(padded>=filter);assert_eq!(out.shape[axis+2],(padded-filter)/stride[axis]+1);
    }
    if out.len==0 {return true;}
    let spatial=out.shape[2..].iter().product::<usize>();
    let chunks=if spatial>=512 {4}else {1};let tiles=spatial.div_ceil(128*chunks);
    let grid=tiles.checked_mul(out.shape[0]).and_then(|v|v.checked_mul(out.shape[1])).and_then(|v|u32::try_from(v).ok());
    let Some(grid)=grid else {return false;};
    let c=client();
    macro_rules! run {($f:ty)=>{forward::launch::<$f,CudaRuntime>(&c,RudaCount::Static(grid,1,1),RudaDim::new_1d(128),
        a.arg(),b.arg(),out.arg(),spatial,tiles,b.shape[0]/groups,stride,padding,dilation,chunks)};}
    unsafe {match out.dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),9=>run!(f64),_=>unreachable!()}}
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);true
}

#[ruda(launch)]
fn transposed<F:Float+RudaElement>(input:&Tensor<F>,weight:&Tensor<F>,output:&mut Tensor<F>,groups:usize,
    #[comptime] stride:Vec<usize>,#[comptime] padding:Vec<usize>,#[comptime] dilation:Vec<usize>)
{
    let p=ABSOLUTE_POS as usize;
    if p<output.len() {
        let mut position=p;let mut starts=Array::<usize>::new(stride.len());let mut sizes=Array::<usize>::new(stride.len());
        let mut coordinates=Array::<i64>::new(stride.len());let mut volume=1usize;
        #[unroll] for reverse in 0usize..stride.len() {
            let axis=comptime!(stride.len()-1-reverse);let coordinate=position%output.shape(axis+2);position/=output.shape(axis+2);
            let center=i64::cast_from(coordinate)+comptime!(padding[axis]) as i64;coordinates[axis]=center;
            let lower=center-i64::cast_from(weight.shape(axis+2)-1)*comptime!(dilation[axis]) as i64;
            let step=comptime!(stride[axis]) as i64;
            let begin=i64::max(0i64,(lower+step-1i64)/step);
            let end=i64::min(i64::cast_from(input.shape(axis+2)),center/step+1i64);
            starts[axis]=usize::cast_from(begin);sizes[axis]=usize::cast_from(i64::max(0i64,end-begin));volume*=sizes[axis];
        }
        let channel=position%output.shape(1);let batch=position/output.shape(1);
        let group=channel/weight.shape(1);let local_channel=channel%weight.shape(1);
        let channels=input.shape(1)/groups;let mut sum=F::cast_from(0);
        for c in 0usize..channels {
            let ic=group*channels+c;
            for tap in 0usize..volume {
                let mut rest=tap;let mut at=batch*input.stride(0)+ic*input.stride(1);
                let mut wt=ic*weight.stride(0)+local_channel*weight.stride(1);let mut valid=true;
                #[unroll] for reverse in 0usize..stride.len() {
                    let axis=comptime!(stride.len()-1-reverse);let coordinate=starts[axis]+rest%sizes[axis];rest/=sizes[axis];
                    let numerator=coordinates[axis]-i64::cast_from(coordinate)*comptime!(stride[axis]) as i64;
                    let step=comptime!(dilation[axis]) as i64;
                    valid=valid && numerator>=0i64 && numerator%step==0i64;
                    at+=coordinate*input.stride(axis+2);
                    wt+=usize::cast_from(numerator/step)*weight.stride(axis+2);
                }
                if valid {sum+=input[at]*weight[wt];}
            }
        }
        output[p]=sum;
    }
}

#[ruda(launch)]
fn weight_gradient<F:Float+RudaElement>(input:&Tensor<F>,gradient:&Tensor<F>,output:&mut Tensor<F>,groups:usize,
    #[comptime] stride:Vec<usize>,#[comptime] padding:Vec<usize>,#[comptime] dilation:Vec<usize>)
{
    let p=ABSOLUTE_POS as usize;
    if p<output.len() {
        let mut position=p;let mut coordinates=Array::<i64>::new(stride.len());let mut volume=1usize;
        #[unroll] for reverse in 0usize..stride.len() {
            let axis=comptime!(stride.len()-1-reverse);let coordinate=position%output.shape(axis+2);position/=output.shape(axis+2);
            coordinates[axis]=i64::cast_from(coordinate)*comptime!(dilation[axis]) as i64-comptime!(padding[axis]) as i64;
            volume*=gradient.shape(axis+2);
        }
        let local_channel=position%output.shape(1);let channel=position/output.shape(1);
        let input_channel=channel/(gradient.shape(1)/groups)*output.shape(1)+local_channel;
        let mut sum=F::cast_from(0);
        for tap in 0usize..volume {
            let mut rest=tap;let mut at=input_channel*input.stride(1);let mut gt=channel*gradient.stride(1);let mut valid=true;
            #[unroll] for reverse in 0usize..stride.len() {
                let axis=comptime!(stride.len()-1-reverse);let coordinate=rest%gradient.shape(axis+2);rest/=gradient.shape(axis+2);
                let source=coordinates[axis]+i64::cast_from(coordinate)*comptime!(stride[axis]) as i64;
                valid=valid && source>=0i64 && source<i64::cast_from(input.shape(axis+2));
                at+=usize::cast_from(source)*input.stride(axis+2);gt+=coordinate*gradient.stride(axis+2);
            }
            if valid {
                for batch in 0usize..input.shape(0) {sum+=input[at+batch*input.stride(0)]*gradient[gt+batch*gradient.stride(0)];}
            }
        }
        output[p]=sum;
    }
}

pub(super) fn backward(op:u32,a:&View,b:&View,out:&View,p:&[i64])->bool {
    let rank=a.shape.len();
    if !(3..=5).contains(&rank) || !out.contiguous() || !matches!(op,1..=3) || ![a,b,out].iter().all(|v|indexable(v)) {return false;}
    let n=rank-2;assert_eq!(p.len(),4*n+1);assert_eq!(b.shape.len(),rank);assert_eq!(out.shape.len(),rank);
    assert_eq!(a.dtype,b.dtype);assert_eq!(a.dtype,out.dtype);
    let groups=usize::try_from(p[4*n]).expect("negative convolution groups");assert!(groups>0);
    let stride:Vec<_>=p[..n].iter().map(|&v|usize::try_from(v).expect("negative stride")).collect();
    let padding:Vec<_>=p[n..2*n].iter().map(|&v|usize::try_from(v).expect("negative padding")).collect();
    let dilation:Vec<_>=p[2*n..3*n].iter().map(|&v|usize::try_from(v).expect("negative dilation")).collect();
    let (input,gradient,weight)=if op==2 {(a,b,out)}else {(out,a,b)};
    assert_eq!(input.shape[0],gradient.shape[0]);assert!(input.shape[1]%groups==0 && gradient.shape[1]%groups==0);
    assert_eq!(weight.shape[0],gradient.shape[1]);assert_eq!(weight.shape[1],input.shape[1]/groups);
    for axis in 0..n {
        assert!(stride[axis]>0 && dilation[axis]>0 && weight.shape[axis+2]>0);
        let filter=dilation[axis].checked_mul(weight.shape[axis+2]-1).and_then(|v|v.checked_add(1)).expect("filter extent overflow");
        if op==3 || op==2 && p[3*n+axis]!=0 {
            let extra=usize::try_from(p[3*n+axis]).expect("negative output padding");
            assert!(extra<stride[axis] || extra<dilation[axis]);
            let expected=gradient.shape[axis+2].checked_sub(1).and_then(|v|v.checked_mul(stride[axis])).and_then(|v|v.checked_add(filter))
                .and_then(|v|v.checked_add(extra)).and_then(|v|v.checked_sub(padding[axis].checked_mul(2)?)).expect("transposed extent overflow");
            assert_eq!(input.shape[axis+2],expected);
        } else {
            let padded=input.shape[axis+2].checked_add(padding[axis].checked_mul(2).expect("padding overflow")).expect("convolution extent overflow");
            assert!(padded>=filter);assert_eq!(gradient.shape[axis+2],(padded-filter)/stride[axis]+1);
        }
    }
    if out.len==0 {return true;}
    let c=client();let count=RudaCount::Static(u32::try_from(out.len.div_ceil(128)).expect("convolution gradient grid overflow"),1,1);
    macro_rules! run {($f:ty)=>{if op==2 {
        weight_gradient::launch::<$f,CudaRuntime>(&c,count,RudaDim::new_1d(128),a.arg(),b.arg(),out.arg(),groups,stride,padding,dilation)
    }else {transposed::launch::<$f,CudaRuntime>(&c,count,RudaDim::new_1d(128),a.arg(),b.arg(),out.arg(),groups,stride,padding,dilation)}};}
    unsafe {match out.dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),9=>run!(f64),_=>unreachable!()}}
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);true
}
