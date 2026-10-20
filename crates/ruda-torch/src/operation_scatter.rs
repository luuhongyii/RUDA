use super::*;

#[ruda(launch)]
fn initialize<E:Numeric+RudaElement>(input:&Tensor<E>,out:&mut Tensor<E>,#[comptime] zero:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        if comptime!(zero) {out[p]=E::cast_from(0);} else {out[p]=input[p];}
    }
}

#[ruda(launch)]
fn check_indices<I:Int+RudaElement>(indices:&Tensor<I>,invalid:&mut Tensor<Atomic<u32>>,bound:u32) {
    let p=ABSOLUTE_POS as usize;
    if p<indices.len() {
        let value=i64::cast_from(indices[p]);
        if value<0i64 || value>=i64::cast_from(bound) {invalid[0].store(1u32);}
    }
}

#[ruda]
fn scalar<E:Numeric>(integer:i64,real:f64,#[comptime] floating:bool)->E {
    if comptime!(floating) {E::cast_from(real)} else {E::cast_from(integer)}
}

#[ruda(launch)]
fn scale<E:Numeric+RudaElement>(input:&Tensor<E>,out:&mut Tensor<E>,integer:i64,real:f64,
    #[comptime] floating:bool,#[comptime] dtype:u32) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let a=input[p];let b=scalar::<E>(integer,real,floating);
        if comptime!(dtype==3) {out[p]=E::cast_from(a!=E::cast_from(0) && b!=E::cast_from(0));}
        else if comptime!(dtype>=4 && dtype<=8) {out[p]=E::cast_from(u64::cast_from(a)*u64::cast_from(b));}
        else {out[p]=a*b;}
    }
}

#[ruda(launch)]
fn scatter_rows<E:Numeric+RudaElement,I:Int+RudaElement>(source:&Tensor<E>,indices:&Tensor<I>,out:&mut Tensor<E>,
    rows:u32,integer:i64,real:f64,#[comptime] floating:bool,#[comptime] axis:usize,
    #[comptime] mode:u32,#[comptime] dtype:u32) {
    let row=ABSOLUTE_POS as usize;
    if row>=rows as usize {terminate!();}
    let mut position=row;let mut base=0usize;let mut source_base=0usize;let mut index_base=0usize;
    let mut d=out.rank();
    while d>0 {
        d-=1;
        if d!=axis {
            let mut extent=out.shape(d);
            if comptime!(mode==0 || mode==1 || mode==4 || mode==5) {extent=indices.shape(d);}
            let coordinate=position%extent;position/=extent;
            base+=coordinate*out.stride(d);
            if comptime!(mode<=4) {source_base+=coordinate*source.stride(d);}
            if comptime!(mode==0 || mode==1 || mode==4 || mode==5) {index_base+=coordinate*indices.stride(d);}
        }
    }
    let mut count=indices.len();
    if comptime!(mode==0 || mode==1 || mode==4 || mode==5) {count=indices.shape(axis);}
    for i in 0..count {
        let mut at=i;
        if comptime!(mode==0 || mode==1 || mode==4 || mode==5) {at=index_base+i*indices.stride(axis);}
        let index=i64::cast_from(indices[at]);
        if index>=0i64 && index<i64::cast_from(out.shape(axis)) {
            let destination=base+usize::cast_from(index)*out.stride(axis);
            let mut value=scalar::<E>(integer,real,floating);
            if comptime!(mode<=4) {value=source[source_base+i*source.stride(axis)];}
            if comptime!(mode==0 || mode==2) {
                let old=out[destination];
                if comptime!(dtype==3) {value=E::cast_from(old!=E::cast_from(0) || value!=E::cast_from(0));}
                else if comptime!(dtype>=4 && dtype<=8) {value=E::cast_from(u64::cast_from(old)+u64::cast_from(value));}
                else {value=old+value;}
            }
            out[destination]=value;
        }
    }
}

fn expanded(value:&View,axis:usize,insert:bool)->View {
    let mut shape=value.shape.clone();
    if insert {shape.insert(axis,1);}
    View::packed(value.handle.clone(),shape,value.dtype)
}

pub(super) fn validate(values:&[View],ints:&[i64],floats:&[f32]) {
    assert_eq!(ints.len(),5);assert!(floats.is_empty());
    let mode=ints[0];assert!((0..=6).contains(&mode) && matches!(ints[3],0|1) && matches!(ints[4],0|1));
    assert_eq!(values.len(),if mode<=3 {4} else {3});
    let out=values.last().unwrap();let q=&values[0];let indices=&values[1];let rank=out.shape.len();
    assert!(rank<=8 && q.dtype<=9 && out.dtype==q.dtype && matches!(indices.dtype,4|5));
    assert!(ints[1]>=0 && (ints[1] as usize)<rank);let axis=ints[1] as usize;
    for value in values {
        super::operation_graph::packed(value);
        assert!(value.shape.len()<=8 && value.shape.iter().all(|&size|size<=u32::MAX as usize));
    }
    if (3..=8).contains(&q.dtype) {assert_eq!(ints[3],0);}
    if mode!=4 {assert_eq!(q.shape,out.shape);assert_eq!(ints[4],0);}
    if mode<=3 {assert_eq!(values[2].dtype,q.dtype);}
    if mode==4 {
        assert_eq!(q.shape,indices.shape);
        let gradient=expanded(q,axis,ints[4]!=0);
        assert_eq!(gradient.shape.len(),rank);
        assert!(gradient.shape.iter().enumerate().all(|(d,&size)|d==axis || size<=out.shape[d]));
    } else if matches!(mode,2|3|6) {
        assert_eq!(indices.shape.len(),1);
        if mode!=6 {
            let source=&values[2];assert_eq!(source.dtype,q.dtype);assert_eq!(source.shape.len(),rank);
            assert!(source.shape.iter().enumerate().all(|(d,&size)|size==if d==axis {indices.len} else {out.shape[d]}));
        }
    } else if indices.len!=0 {
        assert_eq!(indices.shape.len(),rank);
        assert!(indices.shape.iter().enumerate().all(|(d,&size)|d==axis || size<=out.shape[d]));
        if mode<=1 {
            let source=&values[2];assert_eq!(source.dtype,q.dtype);assert_eq!(source.shape.len(),rank);
            assert!(indices.shape.iter().zip(&source.shape).all(|(&a,&b)|a<=b));
        }
    }
}

fn grid(n:usize)->RudaCount {RudaCount::Static(u32::try_from(n.div_ceil(128)).expect("scatter grid overflow"),1,1)}
pub(super) fn checked(kind:u32,values:&[View],ints:&[i64])->bool {
    if kind!=25 || values[1].len==0 {return false;}
    let source=if ints[0]<=3 {&values[2]} else {&values[0]};
    values.last().unwrap().len!=0 || source.len!=0 || ints[0]==5
}

pub(super) fn prepare(c:&ComputeClient<CudaRuntime>,values:&[View],ints:&[i64])
    ->(Vec<PreparedKernel<CudaRuntime>>,Vec<View>) {
    let (mode,axis)=(ints[0] as u32,ints[1] as usize);let out=values.last().unwrap();
    let mut nodes=Vec::new();let mut scratch=Vec::new();
    if checked(25,values,ints) {
        let flag=View::packed(c.empty(4),vec![1],5);let bytes=flag.byte_view();
        nodes.push(kernels::fill_bytes::prepare::<CudaRuntime>(c,grid(4),RudaDim::new_1d(128),bytes.arg(),0u32,0u32));
        macro_rules! check {($i:ty)=>{check_indices::prepare::<$i,CudaRuntime>(c,grid(values[1].len),RudaDim::new_1d(128),values[1].arg(),flag.arg(),out.shape[axis] as u32)};}
        nodes.push(if values[1].dtype==4 {check!(i64)} else {check!(i32)});scratch.push(flag);
    }
    if out.len==0 {return (nodes,scratch);}
    macro_rules! init {($e:ty)=>{initialize::prepare::<$e,CudaRuntime>(c,grid(out.len),RudaDim::new_1d(128),values[0].arg(),out.arg(),mode==4)};}
    nodes.push(match out.dtype {0=>init!(f32),1=>init!(f16),2=>init!(bf16),3|8=>init!(u8),4=>init!(i64),5=>init!(i32),6=>init!(i16),7=>init!(i8),9=>init!(f64),_=>unreachable!()});
    if values[1].len==0 {return (nodes,scratch);}
    let indices=expanded(&values[1],axis,ints[4]!=0);
    let mut source=expanded(if mode<=3 {&values[2]} else {&values[0]},axis,ints[4]!=0);
    let floating=ints[3]!=0;let real=f64::from_bits(ints[2] as u64);
    let unit=if floating {real==1.0} else {ints[2]==1};
    if mode==2 && !unit {
        let target=View::packed(c.empty(source.len.max(1).checked_mul(element_bytes(source.dtype)).expect("scatter scale workspace overflow")),source.shape.clone(),source.dtype);
        if source.len!=0 {
            macro_rules! scale_value {($e:ty)=>{scale::prepare::<$e,CudaRuntime>(c,grid(source.len),RudaDim::new_1d(128),source.arg(),target.arg(),ints[2],real,floating,source.dtype)};}
            nodes.push(match source.dtype {0=>scale_value!(f32),1=>scale_value!(f16),2=>scale_value!(bf16),3|8=>scale_value!(u8),4=>scale_value!(i64),5=>scale_value!(i32),6=>scale_value!(i16),7=>scale_value!(i8),9=>scale_value!(f64),_=>unreachable!()});
        }
        source=View::packed(target.handle.clone(),target.shape.clone(),target.dtype);scratch.push(target);
    }
    let shape=if matches!(mode,2|3|6) {&out.shape} else {&indices.shape};
    let rows=shape.iter().enumerate().filter(|(d,_)|*d!=axis).try_fold(1usize,|n,(_,d)|n.checked_mul(*d))
        .filter(|&n|n<=u32::MAX as usize).expect("scatter row indexing overflow");
    if rows!=0 {
        macro_rules! run {($e:ty,$i:ty)=>{scatter_rows::prepare::<$e,$i,CudaRuntime>(c,grid(rows),RudaDim::new_1d(128),source.arg(),indices.arg(),out.arg(),rows as u32,ints[2],real,floating,axis,mode,out.dtype)};}
        macro_rules! data {($e:ty)=>{if indices.dtype==4 {run!($e,i64)} else {run!($e,i32)}};}
        nodes.push(match out.dtype {0=>data!(f32),1=>data!(f16),2=>data!(bf16),3|8=>data!(u8),4=>data!(i64),5=>data!(i32),6=>data!(i16),7=>data!(i8),9=>data!(f64),_=>unreachable!()});
    }
    (nodes,scratch)
}

pub(super) fn check(c:&ComputeClient<CudaRuntime>,scratch:&[View])->Result<(),String> {
    let bytes=c.read_one(scratch[0].handle.clone()).map_err(|e|e.to_string())?;DOWNLOAD.fetch_add(4,Ordering::Relaxed);
    if bytes.len()!=4 || u32::from_le_bytes(bytes[..4].try_into().unwrap())!=0 {return Err("RUDA graph index out of bounds".into());}
    Ok(())
}
