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

#[ruda(launch)]
fn scatter_overwrite<E:Numeric+RudaElement,I:Int+RudaElement>(input:&Tensor<E>,source:&Tensor<E>,indices:&Tensor<I>,out:&mut Tensor<E>,
    integer:i64,real:f64,#[comptime] floating:bool,#[comptime] axis:usize,#[comptime] mode:u32)
{
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let mut remaining=p;let mut source_base=0usize;let mut index_base=0usize;
        let mut selected=0usize;let mut inside=true;let mut d=out.rank();
        while d>0 {
            d-=1;let coordinate=remaining%out.shape(d);remaining/=out.shape(d);
            if d==axis {selected=coordinate;}
            else {
                if comptime!(mode==1 || mode==5) {
                    if coordinate>=indices.shape(d) {inside=false;}
                    index_base+=coordinate*indices.stride(d);
                }
                if comptime!(mode==1 || mode==3) {source_base+=coordinate*source.stride(d);}
            }
        }
        let mut value=input[p];let mut count=indices.len();
        if comptime!(mode==1 || mode==5) {count=indices.shape(axis);}
        if inside {
            let mut found=false;let mut item=count;
            while item>0 && !found {
                item-=1;let mut index_at=item;
                if comptime!(mode==1 || mode==5) {index_at=index_base+item*indices.stride(axis);}
                if i64::cast_from(indices[index_at])==i64::cast_from(selected) {
                    if comptime!(mode==1 || mode==3) {value=source[source_base+item*source.stride(axis)];}
                    else {value=scalar::<E>(integer,real,floating);}
                    found=true;
                }
            }
        }
        out[p]=value;
    }
}

fn expanded(value:&View,axis:usize,insert:bool)->View {
    let mut shape=value.shape.clone();
    if insert {shape.insert(axis,1);}
    View::packed(value.handle.clone(),shape,value.dtype)
}

#[ruda(launch)]
fn segment_offsets<I:Int+RudaElement>(sorted:&Tensor<I>,offsets:&mut Tensor<u32>) {
    let row=ABSOLUTE_POS as usize;
    if row<offsets.len() {
        let mut low=0usize;let mut high=sorted.len();
        while low<high {
            let middle=low+(high-low)/2usize;
            if i64::cast_from(sorted[middle])<i64::cast_from(row) {low=middle+1usize;}else {high=middle;}
        }
        offsets[row]=low as u32;
    }
}

#[ruda(launch)]
fn ordered_index_update<E:Numeric+RudaElement,V:Size>(input:&Tensor<E>,source:&Tensor<E>,permutation:&Tensor<i64>,offsets:&Tensor<u32>,out:&mut Tensor<E>,alpha_tensor:&Tensor<E>,
    inner:usize,bound:usize,integer:i64,real:f64,#[comptime] floating:bool,#[comptime] scaled:bool,#[comptime] tensor_scalar:bool,
    #[comptime] mode:u32,#[comptime] dtype:u32)
{
    let p=ABSOLUTE_POS as usize;
    if p<out.len()/V::value() {
        let width=inner/V::value();let row=p/width%bound;let outer=p/(bound*width);let column=p%width;
        let base=input.slice(0usize,input.len()).into_vectorized().with_vector_size::<V>();
        let mut value=base[p];let start=offsets[row] as usize;let end=offsets[row+1usize] as usize;
        if comptime!(mode==2) {
            let packed=source.slice(0usize,source.len()).into_vectorized().with_vector_size::<V>();
            for item in start..end {
                let position=permutation[item] as usize;
                let next=packed[(outer*permutation.len()+position)*width+column];
                #[unroll] for lane in 0usize..V::value() {
                    let mut term=next[lane];
                    if comptime!(scaled) {
                        let mut alpha=scalar::<E>(integer,real,floating);
                        if comptime!(tensor_scalar) {alpha=alpha_tensor[0usize];}
                        if comptime!(dtype==3) {term=E::cast_from(term!=E::cast_from(0) && alpha!=E::cast_from(0));}
                        else if comptime!(dtype>=4 && dtype<=8) {term=E::cast_from(u64::cast_from(term)*u64::cast_from(alpha));}
                        else {term=term*alpha;}
                    }
                    if comptime!(dtype==3) {value[lane]=E::cast_from(value[lane]!=E::cast_from(0) || term!=E::cast_from(0));}
                    else if comptime!(dtype>=4 && dtype<=8) {value[lane]=E::cast_from(u64::cast_from(value[lane])+u64::cast_from(term));}
                    else {value[lane]=value[lane]+term;}
                }
            }
        } else if start<end {
            if comptime!(mode==3) {
                let position=permutation[end-1usize] as usize;
                let packed=source.slice(0usize,source.len()).into_vectorized().with_vector_size::<V>();
                value=packed[(outer*permutation.len()+position)*width+column];
            } else {value=Vector::<E,V>::new(scalar::<E>(integer,real,floating));}
        }
        let mut packed=out.slice_mut(0usize,out.len()).into_vectorized().with_vector_size::<V>();packed[p]=value;
    }
}

pub(super) fn prepare_segments(c:&ComputeClient<CudaRuntime>,indices:&View,bound:usize,reuse:&[View])
    ->(Vec<PreparedKernel<CudaRuntime>>,Vec<View>) {
    let allocate=|slot:usize,len:usize,dtype:u32| {
        if let Some(value)=reuse.get(slot) {
            assert_eq!(value.shape,[len]);assert_eq!(value.dtype,dtype);
            View::packed(value.handle.clone(),vec![len],dtype)
        } else {View::packed(c.empty(len.max(1).checked_mul(element_bytes(dtype)).expect("index segment workspace overflow")),vec![len],dtype)}
    };
    let sorted=allocate(0,indices.len,indices.dtype);let permutation=allocate(1,indices.len,4);
    let offsets=allocate(2,bound.checked_add(1).expect("index offsets overflow"),5);
    let sort_values=[View::packed(indices.handle.clone(),vec![indices.len],indices.dtype),
        View::packed(sorted.handle.clone(),vec![indices.len],sorted.dtype),
        View::packed(permutation.handle.clone(),vec![indices.len],4)];
    let (mut nodes,workspace)=operation_sort::prepare_reusing(c,&sort_values,&[1,0,indices.len as i64,0,1,1,0],reuse.get(3..).unwrap_or(&[]));
    macro_rules! bounds {($i:ty)=>{segment_offsets::prepare::<$i,CudaRuntime>(c,grid(bound+1),RudaDim::new_1d(128),sorted.arg(),offsets.arg())};}
    nodes.push(if indices.dtype==4 {bounds!(i64)} else {bounds!(i32)});
    let mut scratch=vec![sorted,permutation,offsets];scratch.extend(workspace);(nodes,scratch)
}

fn prepare_ordered_index_update(c:&ComputeClient<CudaRuntime>,input:&View,source:&View,indices:&View,out:&View,axis:usize,
    mode:u32,integer:i64,real:f64,floating:bool,alpha:Option<&View>,nodes:&mut Vec<PreparedKernel<CudaRuntime>>,scratch:&mut Vec<View>) {
    let bound=out.shape[axis];let (sorting,workspace)=prepare_segments(c,indices,bound,&[]);nodes.extend(sorting);
    let permutation=&workspace[1];let offsets=&workspace[2];
    let inner=out.shape[axis+1..].iter().product::<usize>();
    let width=super::row_cached::memory_vector_size(c,inner,&[input,source,out],16);
    let scaled=mode==2 && (alpha.is_some() || if floating {real!=1.0}else {integer!=1});
    macro_rules! run {($e:ty,$v:ty)=>{ordered_index_update::prepare::<$e,CudaRuntime>(c,grid(out.len/width),RudaDim::new_1d(128),<$v as Size>::value(),
        input.arg(),source.arg(),permutation.arg(),offsets.arg(),out.arg(),alpha.unwrap_or(source).arg(),inner,bound,integer,real,floating,scaled,alpha.is_some(),mode,out.dtype)};}
    macro_rules! data {($e:ty)=>{match width {16=>run!($e,Const<16>),8=>run!($e,Const<8>),4=>run!($e,Const<4>),2=>run!($e,Const<2>),_=>run!($e,Const<1>)}};}
    nodes.push(match out.dtype {0=>data!(f32),1=>data!(f16),2=>data!(bf16),3|8=>data!(u8),4=>data!(i64),5=>data!(i32),6=>data!(i16),7=>data!(i8),9=>data!(f64),_=>unreachable!()});
    scratch.extend(workspace);
}

pub(super) fn try_index_add(source:&View,indices:&View,out:&View,axis:usize)->bool {
    try_scaled_index_add(source,indices,out,axis,None)
}

pub(super) fn try_scaled_index_add(source:&View,indices:&View,out:&View,axis:usize,alpha:Option<&View>)->bool {
    if source.dtype>9 || source.dtype!=out.dtype || !matches!(indices.dtype,4|5) || indices.shape.len()!=1
        || indices.len<256 || axis>=out.shape.len() || out.shape[axis]<8 || source.shape.len()!=out.shape.len()
        || [source,indices,out].iter().any(|v|!v.contiguous() || v.len>u32::MAX as usize)
        || source.handle.memory.clone().binding().id()==out.handle.memory.clone().binding().id()
        || indices.handle.memory.clone().binding().id()==out.handle.memory.clone().binding().id()
    {return false;}
    if alpha.is_some_and(|value|value.len!=1 || value.dtype!=out.dtype || super::graph_fusion::overlaps(value,out)) {return false;}
    for d in 0..out.shape.len() {assert_eq!(source.shape[d],if d==axis {indices.len}else {out.shape[d]});}
    if out.len==0 {return true;}
    let c=client().fixed_execution_queue();let mut nodes=Vec::new();let mut scratch=Vec::new();
    prepare_ordered_index_update(&c,out,source,indices,out,axis,2,1,1.0,false,alpha,&mut nodes,&mut scratch);
    let launches=nodes.len();
    for node in nodes {let (task,count,args,queue)=node.into_parts();queue.launch(task,count,args);}
    LAUNCHES.fetch_add(launches as u64,Ordering::Relaxed);finish_dispatch(&c);true
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
    let count=if matches!(mode,1|5) {values[1].shape[axis]} else {values[1].len};
    if matches!(mode,1|3|5|6) && (1..=16).contains(&count) {
        let source=if mode<=3 {&values[2]} else {&values[0]};
        macro_rules! data {($e:ty)=>{{
            macro_rules! run {($i:ty)=>{scatter_overwrite::prepare::<$e,$i,CudaRuntime>(c,grid(out.len),RudaDim::new_1d(128),
                values[0].arg(),source.arg(),values[1].arg(),out.arg(),ints[2],f64::from_bits(ints[2] as u64),ints[3]!=0,axis,mode)};}
            if values[1].dtype==4 {run!(i64)}else {run!(i32)}
        }};}
        nodes.push(match out.dtype {0=>data!(f32),1=>data!(f16),2=>data!(bf16),3|8=>data!(u8),4=>data!(i64),5=>data!(i32),6=>data!(i16),7=>data!(i8),9=>data!(f64),_=>unreachable!()});
        return (nodes,scratch);
    }
    if matches!(mode,2|3|6) && values[1].len>=256 && out.shape[axis]>=8 {
        let source=if mode<=3 {&values[2]}else {&values[0]};
        prepare_ordered_index_update(c,&values[0],source,&values[1],out,axis,mode,ints[2],f64::from_bits(ints[2] as u64),ints[3]!=0,None,&mut nodes,&mut scratch);
        return (nodes,scratch);
    }
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
