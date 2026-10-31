use super::*;

#[ruda]
fn mapped<E:Numeric>(input:&Tensor<E>,out:&Tensor<E>,p:usize,
    #[comptime] parameters:Vec<i64>,#[comptime] mode:u32,#[comptime] rank:usize,
    #[comptime] input_rank:usize)->(usize,bool)
{
    let mut source=0usize;let mut inside=true;let mut rest=p;
    if comptime!(mode==1 && parameters[3]!=0) {
        source=p/comptime!(parameters[2] as usize);
    } else if comptime!(mode==5 && parameters[1]!=0) {
        source=(p+input.len()-comptime!(parameters[2] as usize))%input.len();
    } else {
        #[unroll]
        for axis in 0..rank {
            let d=comptime!(rank-axis-1);
            let coordinate=rest%out.shape(d);rest/=out.shape(d);
            if comptime!(mode==0) {
                if comptime!(d>=rank-input_rank) {
                    let sd=comptime!(d-(rank-input_rank));
                    source+=(coordinate%input.shape(sd))*input.stride(sd);
                }
            } else if comptime!(mode==1) {
                let mut mapped=coordinate;
                if comptime!(d==parameters[1] as usize) {mapped/=comptime!(parameters[2] as usize);}
                source+=mapped*input.stride(d);
            } else if comptime!(mode==2) {
                let mapped=i64::cast_from(coordinate)-comptime!(parameters[3+2*d]);
                if mapped<0i64 || mapped>=i64::cast_from(input.shape(d)) {inside=false;}
                else {source+=usize::cast_from(mapped)*input.stride(d);}
            } else if comptime!(mode==3) {
                let mut mapped=coordinate;
                if comptime!(d==parameters[1] as usize) {
                    mapped=comptime!(parameters[2] as usize)+coordinate*comptime!(parameters[4] as usize);
                }
                source+=mapped*input.stride(d);
            } else if comptime!(mode==4) {
                let mut mapped=coordinate;
                if comptime!(d==parameters[1] as usize) {
                    let delta=i64::cast_from(coordinate)-comptime!(parameters[2]);
                    if delta<0i64 || i64::cast_from(coordinate)>=comptime!(parameters[3]) {inside=false;}
                    else if delta%comptime!(parameters[4])!=0i64 {inside=false;}
                    else {mapped=usize::cast_from(delta/comptime!(parameters[4]));}
                }
                source+=mapped*input.stride(d);
            } else {
                let shift=comptime!(parameters[2+d] as usize);
                let mapped=(coordinate+input.shape(d)-shift)%input.shape(d);
                source+=mapped*input.stride(d);
            }
        }
    }
    (source,inside)
}

#[ruda(launch)]
fn transform<E:Numeric+RudaElement>(input:&Tensor<E>,out:&mut Tensor<E>,integer_fill:i64,float_fill:f32,
    #[comptime] parameters:Vec<i64>,#[comptime] mode:u32,#[comptime] rank:usize,
    #[comptime] input_rank:usize,#[comptime] integral:bool)
{
    let p=ABSOLUTE_POS as usize;
    if p>=out.len() {terminate!();}
    let source=mapped(input,out,p,parameters,mode,rank,input_rank);
    if source.1 {out[p]=input[source.0];}
    else if comptime!(mode==2 && integral) {out[p]=E::cast_from(integer_fill);}
    else if comptime!(mode==2) {out[p]=E::cast_from(float_fill);}
    else {out[p]=E::cast_from(0);}
}

#[ruda(launch)]
fn transform_vector<E:Numeric+RudaElement>(input:&Tensor<E>,out:&mut Tensor<E>,integer_fill:i64,float_fill:f32,
    #[comptime] parameters:Vec<i64>,#[comptime] mode:u32,#[comptime] rank:usize,
    #[comptime] input_rank:usize,#[comptime] integral:bool)
{
    let p=ABSOLUTE_POS as usize;
    if p<out.len()/4 {
        let source=mapped(input,out,p*4usize,parameters,mode,rank,input_rank);
        let mut value=Vector::<E,Const<4>>::new(E::cast_from(0));
        if source.1 {value=input.slice(0usize,input.len()).into_vectorized().with_vector_size::<Const<4>>()[source.0/4];}
        else if comptime!(mode==2 && integral) {value=Vector::new(E::cast_from(integer_fill));}
        else if comptime!(mode==2) {value=Vector::new(E::cast_from(float_fill));}
        let mut packed=out.slice_mut(0usize,out.len()).into_vectorized().with_vector_size::<Const<4>>();packed[p]=value;
    }
}

fn vectorizable(input:&View,out:&View,ints:&[i64])->bool {
    if input.shape.is_empty() || out.shape.is_empty() || [input,out].iter().any(|v|!v.contiguous()
        || v.shape.last().unwrap()%4!=0 || v.handle.offset_start.unwrap_or(0)%(4*element_bytes(v.dtype)) as u64!=0) {return false;}
    let last=input.shape.len()-1;
    match ints[0] {
        0=>true,
        1=>ints[3]==0 && ints[1] as usize!=last,
        2=>ints[3+2*last]%4==0 && ints[4+2*last]%4==0,
        3|4=>ints[1] as usize!=last || ints[4]==1 && ints[2]%4==0 && ints[3]%4==0,
        5=>ints[if ints[1]!=0 {2}else {2+last}]%4==0,
        _=>false,
    }
}

fn dimension(value:i128)->usize {
    assert!((0..=u32::MAX as i128).contains(&value),"transform dimension overflow");value as usize
}

pub(super) fn validate(values:&[View],ints:&[i64],floats:&[f32]) {
    assert_eq!(values.len(),2);assert!(!ints.is_empty());
    let (input,out)=(&values[0],&values[1]);let rank=input.shape.len();
    assert!(rank<=8 && out.shape.len()<=8);assert_eq!(input.dtype,out.dtype);
    assert!(input.dtype<=9 || matches!(input.dtype,12|13));
    assert!(values.iter().flat_map(|v|&v.shape).all(|&d|d<=u32::MAX as usize));
    let mode=ints[0];assert!((0..=5).contains(&mode));
    assert_eq!(floats.len(),if mode==2 {1}else {0});
    match mode {
        0=>{
            assert_eq!(ints.len(),out.shape.len()+1);assert!(out.shape.len()>=rank);
            let leading=out.shape.len()-rank;
            for d in 0..out.shape.len() {
                assert!(ints[d+1]>=0);
                let size=if d<leading {1}else {input.shape[d-leading]};
                assert_eq!(out.shape[d],dimension(size as i128*ints[d+1] as i128));
            }
        },
        1=>{
            assert_eq!(ints.len(),4);assert!(ints[2]>=0 && matches!(ints[3],0|1));
            if ints[3]!=0 {assert_eq!(out.shape,vec![dimension(input.len as i128*ints[2] as i128)]);}
            else {
                assert!(ints[1]>=0 && (ints[1] as usize)<rank);let axis=ints[1] as usize;
                let mut shape=input.shape.clone();shape[axis]=dimension(shape[axis] as i128*ints[2] as i128);
                assert_eq!(out.shape,shape);
            }
        },
        2=>{
            assert_eq!(ints.len(),3+2*rank);assert!((0..=rank as i64).contains(&ints[2]));
            assert!(input.dtype<=9);assert_eq!(out.shape.len(),rank);
            let positive=ints[3..].iter().any(|&p|p>0);
            for d in 0..rank {
                let (before,after)=(ints[3+2*d] as i128,ints[4+2*d] as i128);
                if d<rank-ints[2] as usize {assert_eq!((before,after),(0,0));}
                let remaining=input.shape[d] as i128+before.min(0)+after.min(0);
                assert!(remaining>=0,"negative padding exceeds input");
                let size=dimension(input.shape[d] as i128+before+after);
                if positive && d>=rank-ints[2] as usize {assert!(size>0,"positive padding requires positive padded dimensions");}
                assert_eq!(out.shape[d],size);
            }
        },
        3|4=>{
            assert_eq!(ints.len(),5);assert!(ints[1]>=0 && (ints[1] as usize)<rank && ints[4]>0);
            let axis=ints[1] as usize;let (full,part)=if mode==3 {(input,out)}else {(out,input)};
            assert_eq!(full.shape.len(),part.shape.len());
            assert!((0..=full.shape[axis] as i64).contains(&ints[2]) && (0..=full.shape[axis] as i64).contains(&ints[3]));
            let length=(ints[3] as i128-ints[2] as i128).max(0);
            let mut shape=full.shape.clone();shape[axis]=dimension((length+ints[4] as i128-1)/ints[4] as i128);
            assert_eq!(part.shape,shape);
        },
        5=>{
            assert!(ints.len()>=3 && matches!(ints[1],0|1));assert_eq!(input.shape,out.shape);
            if ints[1]!=0 {assert_eq!(ints.len(),3);assert!(ints[2]>=0 && (ints[2] as usize)<input.len.max(1));}
            else {
                assert_eq!(ints.len(),rank+2);
                for d in 0..rank {assert!(ints[2+d]>=0 && (ints[2+d] as usize)<input.shape[d].max(1));}
            }
        },
        _=>unreachable!(),
    }
}

pub(super) fn prepare(c:&ComputeClient<CudaRuntime>,values:&[View],ints:&[i64],floats:&[f32])->Vec<PreparedKernel<CudaRuntime>> {
    let (input,out)=(&values[0],&values[1]);if out.len==0 {return Vec::new();}
    let mode=ints[0] as u32;let fill=if mode==2 {ints[1]}else {0};
    let scalar=if mode==2 {floats[0]}else {0.0};
    if vectorizable(input,out,ints) {
        let count=RudaCount::Static(u32::try_from((out.len/4).div_ceil(128)).expect("vector transform grid overflow"),1,1);
        macro_rules! run {($e:ty)=>{transform_vector::prepare::<$e,CudaRuntime>(c,count,RudaDim::new_1d(128),input.arg(),out.arg(),
            fill,scalar,ints.to_vec(),mode,out.shape.len(),input.shape.len(),(3..=8).contains(&out.dtype))};}
        return vec![match out.dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),3|8|12|13=>run!(u8),4=>run!(i64),
            5=>run!(i32),6=>run!(i16),7=>run!(i8),9=>run!(f64),_=>unreachable!()}];
    }
    let count=RudaCount::Static(u32::try_from(out.len.div_ceil(128)).expect("transform grid overflow"),1,1);
    macro_rules! run {($e:ty)=>{transform::prepare::<$e,CudaRuntime>(c,count,RudaDim::new_1d(128),input.arg(),out.arg(),
        fill,scalar,ints.to_vec(),mode,out.shape.len(),input.shape.len(),(3..=8).contains(&out.dtype))};}
    vec![match out.dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),3|8|12|13=>run!(u8),4=>run!(i64),
        5=>run!(i32),6=>run!(i16),7=>run!(i8),9=>run!(f64),_=>unreachable!()}]
}
