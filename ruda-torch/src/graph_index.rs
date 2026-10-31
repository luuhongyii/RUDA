use super::*;
use super::graph_contract::NodeSpec;

#[ruda(launch)]
fn to_bool<E:Numeric+RudaElement>(input:&Tensor<E>,output:&mut Tensor<u8>) {
    let p=ABSOLUTE_POS as usize;
    if p<output.len() {output[p]=u8::cast_from(input[p]!=E::cast_from(0));}
}

#[ruda(launch)]
fn indexed<E:Numeric+RudaElement,I:Int+RudaElement>(
    input:&Tensor<E>,indices:&Tensor<I>,output:&mut Tensor<E>,invalid:&mut Tensor<Atomic<u32>>,
    #[comptime] axis:usize,#[comptime] operation:u32,
) {
    let p=ABSOLUTE_POS as usize;
    if p<output.len() {
        let mut source=0usize;let mut at=0usize;
        if comptime!(operation==136) {
            let width=input.shape(1);at=p/width;source=p%width;
        } else {
            let mut rest=p;let mut d=output.rank();
            while d>0 {
                d-=1;let coordinate=rest%output.shape(d);rest/=output.shape(d);
                if d==axis {at=coordinate;} else {source+=coordinate*input.stride(d);}
            }
            if comptime!(operation==135) {at=p;}
        }
        let index=i64::cast_from(indices[at]);let bound=i64::cast_from(input.shape(axis));
        if index<0i64 || index>=bound {
            invalid[0].store(1u32);output[p]=E::cast_from(0);
        } else {output[p]=input[source+usize::cast_from(index)*input.stride(axis)];}
    }
}

#[ruda(launch)]
fn frequencies<I:Int+RudaElement>(indices:&Tensor<I>,counts:&mut Tensor<Atomic<u32>>,invalid:&mut Tensor<Atomic<u32>>) {
    let p=ABSOLUTE_POS as usize;
    if p<indices.len() {
        let value=i64::cast_from(indices[p]);
        if value<0i64 || value>=i64::cast_from(counts.len()) {invalid[0].store(1u32);}
        else {counts[usize::cast_from(value)].fetch_add(1u32);}
    }
}

#[ruda(launch)]
fn indexed_vector<E:Numeric+RudaElement,I:Int+RudaElement>(input:&Tensor<E>,indices:&Tensor<I>,output:&mut Tensor<E>,
    invalid:&mut Tensor<Atomic<u32>>,inner:usize,bound:usize)
{
    let p=ABSOLUTE_POS as usize;
    if p<output.len()/4 {
        let row=p/(inner/4);let column=p%(inner/4);
        let index=i64::cast_from(indices[row%indices.len()]);
        let mut value=Vector::<E,Const<4>>::new(E::cast_from(0));
        if index<0i64 || index>=i64::cast_from(bound) {invalid[0].store(1u32);}
        else {
            let source=(row/indices.len()*bound+usize::cast_from(index))*(inner/4)+column;
            let packed=input.slice(0usize,input.len()).into_vectorized().with_vector_size::<Const<4>>();
            value=packed[source];
        }
        let mut packed=output.slice_mut(0usize,output.len()).into_vectorized().with_vector_size::<Const<4>>();
        packed[p]=value;
    }
}

#[ruda(launch)]
fn selected_direct<E:Numeric+RudaElement,I:Int+RudaElement,V:Size>(input:&Tensor<E>,indices:&Tensor<I>,output:&mut Tensor<E>,
    inner:usize,bound:usize,#[comptime] reverse:bool)
{
    let p=ABSOLUTE_POS as usize;
    let mut length=output.len();if comptime!(reverse) {length=input.len();}
    if p<length/V::value() {
        let row=p/(inner/V::value());let column=p%(inner/V::value());
        let index=usize::cast_from(indices[row%indices.len()]);
        let source=(row/indices.len()*bound+index)*(inner/V::value())+column;
        let packed=input.slice(0usize,input.len()).into_vectorized().with_vector_size::<V>();
        let mut target=output.slice_mut(0usize,output.len()).into_vectorized().with_vector_size::<V>();
        if comptime!(reverse) {target[source]=packed[p];}else {target[p]=packed[source];}
    }
}

pub(super) fn try_select(a:&View,indices:&View,out:&View,axis:usize)->bool {
    try_selected_direct(a,indices,out,axis,false)
}

pub(super) fn try_copy(a:&View,indices:&View,out:&View,axis:usize)->bool {
    try_selected_direct(a,indices,out,axis,true)
}

fn try_selected_direct(a:&View,indices:&View,out:&View,axis:usize,reverse:bool)->bool {
    let indexed=if reverse {out}else {a};let selected=if reverse {a}else {out};
    if axis>=indexed.shape.len() || a.dtype!=out.dtype || a.dtype>9 || indices.shape.len()!=1
        || !matches!(indices.dtype,4|5) || ![a,indices,out].iter().all(|v|v.contiguous()) {return false;}
    if super::graph_fusion::overlaps(a,out) || super::graph_fusion::overlaps(indices,out) {return false;}
    let mut shape=indexed.shape.clone();shape[axis]=indices.len;
    assert_eq!(shape,selected.shape);
    if selected.len==0 {return true;}
    let inner=indexed.shape[axis+1..].iter().product::<usize>();
    let vector=inner%4==0 && [a,out].iter().all(|v|v.handle.offset_start.unwrap_or(0)%(4*element_bytes(v.dtype)) as u64==0);
    let width=if vector {4}else {1};let c=client();
    macro_rules! data {($e:ty)=>{{
        macro_rules! run {($i:ty,$v:ty)=>{selected_direct::prepare::<$e,$i,$v,CudaRuntime>(&c,grid(selected.len/width),RudaDim::new_1d(128),
            a.arg(),indices.arg(),out.arg(),inner,indexed.shape[axis],reverse)};}
        if indices.dtype==4 {if vector {run!(i64,Const<4>)}else {run!(i64,Const<1>)}}
        else if vector {run!(i32,Const<4>)}else {run!(i32,Const<1>)}
    }};}
    let prepared=match out.dtype {0=>data!(f32),1=>data!(f16),2=>data!(bf16),3|8=>data!(u8),4=>data!(i64),5=>data!(i32),6=>data!(i16),7=>data!(i8),9=>data!(f64),_=>unreachable!()};
    let (task,count,args,origin)=prepared.into_parts();origin.launch(task,count,args);
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);true
}

#[ruda(launch)]
fn bounds_check<I:Int+RudaElement>(indices:&Tensor<I>,invalid:&mut Tensor<Atomic<u32>>,lower:i64,upper:i64) {
    let p=ABSOLUTE_POS as usize;
    if p<indices.len() {
        let value=i64::cast_from(indices[crate::scientific::address(indices,p)]);
        if value<lower || value>=upper {invalid[0].store(1u32);}
    }
}

pub(super) fn check_bounds(indices:&View,lower:i64,upper:i64) {
    let c=client();let flag=View::packed(c.empty(4),vec![1],5);let bytes=flag.byte_view();
    let init=kernels::fill_bytes::prepare::<CudaRuntime>(&c,RudaCount::Static(1,1,1),RudaDim::new_1d(128),bytes.arg(),0u32,0u32);
    let (task,count,args,origin)=init.into_parts();origin.launch(task,count,args);
    macro_rules! run {($i:ty)=>{bounds_check::prepare::<$i,CudaRuntime>(&c,grid(indices.len),RudaDim::new_1d(128),indices.arg(),flag.arg(),lower,upper)};}
    let check=if indices.dtype==4 {run!(i64)}else {run!(i32)};
    let (task,count,args,origin)=check.into_parts();origin.launch(task,count,args);
    sync(&c);LAUNCHES.fetch_add(2,Ordering::Relaxed);
    let bytes=c.read_one(flag.handle).expect("RUDA index status readback failed");
    DOWNLOAD.fetch_add(4,Ordering::Relaxed);
    assert_eq!(bytes.len(),4);
    assert_eq!(u32::from_le_bytes(bytes[..4].try_into().unwrap()),0,"RUDA index out of bounds");
}

#[ruda(launch)]
fn indexed_backward<E:Float+RudaElement,I:Int+RudaElement>(
    gradient:&Tensor<E>,indices:&Tensor<I>,output:&mut Tensor<Atomic<f32>>,counts:&Tensor<u32>,
    invalid:&mut Tensor<Atomic<u32>>,padding:i64,#[comptime] axis:usize,#[comptime] operation:u32,
) {
    let p=ABSOLUTE_POS as usize;
    if p<gradient.len() {
        let mut destination=0usize;let mut at=p;
        if comptime!(operation==137 || operation==138) {
            let width=output.shape(1);at=p/width;destination=p%width;
        } else {
            let mut rest=p;let mut d=gradient.rank();
            while d>0 {
                d-=1;let coordinate=rest%gradient.shape(d);rest/=gradient.shape(d);
                if d!=axis {destination+=coordinate*output.stride(d);}
            }
        }
        let index=i64::cast_from(indices[at]);let bound=i64::cast_from(output.shape(axis));
        if index<0i64 || index>=bound {invalid[0].store(1u32);}
        else {
            let mut include=true;
            if comptime!(operation==137 || operation==138) {include=index!=padding;}
            if include {
                let mut value=f32::cast_from(gradient[p]);
                if comptime!(operation==138) {value/=f32::cast_from(counts[usize::cast_from(index)]);}
                output[destination+usize::cast_from(index)*output.stride(axis)].fetch_add(value);
            }
        }
    }
}

#[ruda(launch)]
fn embedding_segmented_backward<E:Float+RudaElement,V:Size>(gradient:&Tensor<E>,permutation:&Tensor<i64>,offsets:&Tensor<u32>,
    output:&mut Tensor<E>,padding:i64,#[comptime] frequency:bool)
{
    let p=ABSOLUTE_POS as usize;
    if p<output.len()/V::value() {
        let width=output.shape(1)/V::value();let row=p/width;let column=p%width;
        let mut value=Vector::<f32,V>::new(0.0f32);
        if i64::cast_from(row)!=padding {
            let start=offsets[row] as usize;let end=offsets[row+1usize] as usize;
            let packed=gradient.slice(0usize,gradient.len()).into_vectorized().with_vector_size::<V>();
            for item in start..end {
                let position=permutation[item] as usize;let next=packed[position*width+column];
                #[unroll] for lane in 0usize..V::value() {
                    let mut term=f32::cast_from(next[lane]);
                    if comptime!(frequency) {term/=f32::cast_from(end-start);}
                    value[lane]+=term;
                }
            }
        }
        let mut result=Vector::<E,V>::empty();
        #[unroll] for lane in 0usize..V::value() {result[lane]=E::cast_from(value[lane]);}
        let mut packed=output.slice_mut(0usize,output.len()).into_vectorized().with_vector_size::<V>();packed[p]=result;
    }
}

pub(super) fn cat(a:&View,b:&View,out:&View,axis:usize) {
    let rank=a.shape.len();assert!(axis<rank);
    assert_eq!(b.shape.len(),rank);assert_eq!(out.shape.len(),rank);
    assert_eq!(a.dtype,b.dtype);assert_eq!(a.dtype,out.dtype);
    for d in 0..rank {
        if d==axis {assert_eq!(a.shape[d].checked_add(b.shape[d]).expect("cat shape overflow"),out.shape[d]);}
        else {assert_eq!(a.shape[d],b.shape[d]);assert_eq!(a.shape[d],out.shape[d]);}
    }
    for value in [a,b,out] {scientific::indexable(value);}
    let mut stride=1usize;
    for (&size,&actual) in out.shape.iter().zip(&out.strides).rev() {
        if size>1 {assert_eq!(actual,stride);}
        stride=stride.checked_mul(size).expect("cat shape overflow");
    }
    if out.len==0 {return;}
    let c=client();
    macro_rules! run {($e:ty)=>{transform::prepare::<$e,CudaRuntime>(&c,grid(out.len),RudaDim::new_1d(128),a.arg(),b.arg(),out.arg(),142,axis,0)};}
    let prepared=if let Some(prepared)=prepare_vector_cat(&c,a,b,out,axis) {prepared}
        else {match out.dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),_=>panic!("direct cat requires float storage")}};
    let (task,count,args,origin)=prepared.into_parts();origin.launch(task,count,args);
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);
}

fn grid(n:usize)->RudaCount {RudaCount::Static(u32::try_from(n.div_ceil(128)).expect("graph indexing grid overflow"),1,1)}

fn prepare_vector_cat(c:&ComputeClient<CudaRuntime>,a:&View,b:&View,out:&View,axis:usize)->Option<PreparedKernel<CudaRuntime>> {
    if out.dtype>2 || out.len<16384 || out.len>i32::MAX as usize || !out.contiguous()
        || !a.vector_rows(true) || !b.vector_rows(true) || !out.vector_rows(false) {return None;}
    Some(strided_vector::cat::prepare::<CudaRuntime>(c,grid(out.len/4),RudaDim::new_1d(128),
        a.arg(),b.arg(),out.arg(),strided_vector::shape::<CudaRuntime>(&out.shape),axis,
        a.strides.last()==Some(&0),b.strides.last()==Some(&0),strided_vector::storage(out.dtype)))
}

#[ruda]
pub(super) fn broadcast_index<E:Numeric,O:Numeric>(input:&Tensor<E>,out:&Tensor<O>,position:usize)->usize {
    let mut rest=position;let mut index=0usize;let mut dim=out.rank();
    while dim>0 {
        dim-=1;let c=rest%out.shape(dim);rest/=out.shape(dim);
        if dim+input.rank()>=out.rank() {
            let source=dim+input.rank()-out.rank();
            if input.shape(source)!=1 {index+=c*input.stride(source);}
        }
    }
    index
}

#[ruda(launch)]
fn masked_fill<E:Numeric+RudaElement>(input:&Tensor<E>,mask:&Tensor<u8>,out:&mut Tensor<E>,value:f32,#[comptime] boolean:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        if mask[broadcast_index(mask,out,p)]!=0u8 {
            if comptime!(boolean) {out[p]=E::cast_from(value!=0.0f32);} else {out[p]=E::cast_from(value);}
        } else {out[p]=input[broadcast_index(input,out,p)];}
    }
}

#[ruda(launch)]
fn where_tensor<E:Numeric+RudaElement>(mask:&Tensor<u8>,yes:&Tensor<E>,no:&Tensor<E>,out:&mut Tensor<E>) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        if mask[broadcast_index(mask,out,p)]!=0u8 {out[p]=yes[broadcast_index(yes,out,p)];}
        else {out[p]=no[broadcast_index(no,out,p)];}
    }
}

#[ruda(launch)]
fn compare<E:Numeric+RudaElement>(a:&Tensor<E>,b:&Tensor<E>,out:&mut Tensor<u8>,scalar:f32,#[comptime] operation:u32) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let x=a[broadcast_index(a,out,p)];let mut y=E::cast_from(scalar);
        if comptime!(operation<154) {y=b[broadcast_index(b,out,p)];}
        let mut value=false;
        if comptime!(operation==148 || operation==154) {value=x==y;}
        else if comptime!(operation==149 || operation==155) {value=x!=y;}
        else if comptime!(operation==150 || operation==156) {value=x<y;}
        else if comptime!(operation==151 || operation==157) {value=x<=y;}
        else if comptime!(operation==152 || operation==158) {value=x>y;}
        else {value=x>=y;}
        out[p]=u8::cast_from(value);
    }
}

#[ruda(launch)]
fn transform<E:Numeric+RudaElement>(a:&Tensor<E>,b:&Tensor<E>,out:&mut Tensor<E>,
    #[comptime] operation:u32,#[comptime] axis:usize,start:usize) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let mut rest=p;let mut source=0usize;let mut second=0usize;let mut selected=true;
        if comptime!(operation==141) {
            let mut d=a.rank();let mut output_dim=a.rank()-1;
            while d>0 {
                d-=1;
                if d==axis {source+=start*a.stride(d);}
                else {output_dim-=1;source+=(rest%out.shape(output_dim))*a.stride(d);rest/=out.shape(output_dim);}
            }
            out[p]=a[source];
        } else {
            let mut d=out.rank();
            while d>0 {
                d-=1;let coordinate=rest%out.shape(d);rest/=out.shape(d);
                if comptime!(operation==140) {
                    let mut c=coordinate;if d==axis {c+=start;}source+=c*a.stride(d);
                } else if comptime!(operation==142) {
                    let mut c=coordinate;
                    if d==axis && c>=a.shape(d) {selected=false;c-=a.shape(d);}
                    source+=coordinate*a.stride(d);second+=c*b.stride(d);
                } else if comptime!(operation==144) {
                    if d==axis {if coordinate!=start {selected=false;}}
                    else {let mut input_dim=d;if d>axis {input_dim-=1;}source+=coordinate*a.stride(input_dim);}
                } else if comptime!(operation==143) {
                    let mut c=coordinate;
                    if d==axis {if c<start || c>=start+a.shape(d) {selected=false;}else {c-=start;}}
                    source+=c*a.stride(d);
                } else {
                    source+=coordinate*a.stride(d);let mut c=coordinate;
                    if d==axis {if c<start || c>=start+b.shape(d) {selected=false;}else {c-=start;}}
                    second+=c*b.stride(d);
                }
            }
            if comptime!(operation==140) {out[p]=a[source];}
            else if comptime!(operation==142) {if selected {out[p]=a[source];} else {out[p]=b[second];}}
            else if comptime!(operation==143 || operation==144) {if selected {out[p]=a[source];} else {out[p]=E::cast_from(0);}}
            else {if selected {out[p]=b[second];} else {out[p]=a[source];}}
        }
    }
}

pub(super) fn prepare_layout(client:&ComputeClient<CudaRuntime>,a:&View,out:&View,n:&NodeSpec)->Option<PreparedKernel<CudaRuntime>> {
    if matches!(n.op,194|195) {return Some(super::segment_scan::prepare(client,a,out,n.op==195));}
    if n.op==193 {return Some(super::quantized::prepare_nibbles(client,a,out));}
    if !matches!(n.op,0|112..=115) {return None;}
    if n.op==114 && a.dtype!=out.dtype && (matches!(a.dtype,12|13|15) || matches!(out.dtype,12|13|15)) {
        return Some(super::quantized::prepare_convert(client,a,out));
    }
    let tiled = if n.op==113 {super::layout_tiled::prepare_permute(client,a,out,n.scalar as u32)}
        else if n.op!=115 {super::layout_tiled::prepare_copy(client,a,out)} else {None};
    if tiled.is_some() {return tiled;}
    macro_rules! input {($e:ty)=>{{
        if n.op==113 {unsafe{kernels::graph_permute::prepare::<$e,CudaRuntime>(client,grid(out.len),RudaDim::new_1d(128),a.arg(),out.arg(),n.scalar as u32)}}
        else if n.op==115 {unsafe{kernels::graph_expand::prepare::<$e,CudaRuntime>(client,grid(out.len),RudaDim::new_1d(128),a.arg(),out.arg())}}
        else if out.dtype==3 {unsafe{to_bool::prepare::<$e,CudaRuntime>(client,grid(out.len),RudaDim::new_1d(128),a.arg(),out.arg())}}
        else {
            macro_rules! output {($o:ty)=>{unsafe{kernels::convert::prepare::<$e,$o,CudaRuntime>(client,grid(out.len),RudaDim::new_1d(128),a.arg(),out.arg())}};}
            match out.dtype {0=>output!(f32),1=>output!(f16),2=>output!(bf16),4=>output!(i64),5=>output!(i32),6=>output!(i16),7=>output!(i8),8|12|13|15=>output!(u8),_=>unreachable!()}
        }
    }};}
    Some(match a.dtype {0=>input!(f32),1=>input!(f16),2=>input!(bf16),3|8|12|13|15=>input!(u8),4=>input!(i64),5=>input!(i32),6=>input!(i16),7=>input!(i8),_=>unreachable!()})
}

enum BackwardBuffers {Atomic{accumulation:View,counts:Option<View>},Segmented}
pub(super) struct IndexBuffers {invalid:Option<View>,backward:Vec<Option<BackwardBuffers>>}
impl IndexBuffers {
    pub fn enabled(&self)->bool {self.invalid.is_some()}
    pub fn error_flag(&self)->Option<&View> {self.invalid.as_ref()}
    pub fn new(client:&ComputeClient<CudaRuntime>,tensors:&[View],nodes:&[NodeSpec],inputs:usize)->Self {
        let invalid=nodes.iter().any(|n|(134..=139).contains(&n.op)).then(||View::packed(client.empty(4),vec![1],5));
        let backward=nodes.iter().enumerate().map(|(index,node)| {
            if !(137..=139).contains(&node.op) {return None;}
            let out=&tensors[inputs+index];
            let a=&tensors[node.a as usize];let b=&tensors[node.b as usize];
            if matches!(node.op,137|138) && b.len>=256 && out.shape[0]>=8 && out.shape[1]>=4
                && a.dtype==out.dtype && [a,b,out].iter().all(|v|v.contiguous()) {
                return Some(BackwardBuffers::Segmented);
            }
            let accumulation=View::packed(if out.dtype==0 {out.handle.clone()} else {client.empty(out.len.checked_mul(4).expect("gradient workspace overflow").max(4))},out.shape.clone(),0);
            let counts=(node.op==138).then(||View::packed(client.empty(out.shape[0].checked_mul(4).expect("frequency workspace overflow").max(4)),vec![out.shape[0]],5));
            Some(BackwardBuffers::Atomic{accumulation,counts})
        }).collect();
        Self{invalid,backward}
    }
    fn zero(client:&ComputeClient<CudaRuntime>,value:&View)->PreparedKernel<CudaRuntime> {
        let bytes=value.byte_view();
        unsafe{kernels::fill_bytes::prepare::<CudaRuntime>(client,grid(bytes.len),RudaDim::new_1d(128),bytes.arg(),0u32,0u32)}
    }
    pub fn preamble(&self,client:&ComputeClient<CudaRuntime>)->Vec<PreparedKernel<CudaRuntime>> {
        self.invalid.as_ref().map(|value|vec![Self::zero(client,value)]).unwrap_or_default()
    }
    pub fn check(&self,client:&ComputeClient<CudaRuntime>)->Result<(),String> {
        if let Some(invalid)=&self.invalid {
            let bytes=client.read_one(invalid.handle.clone()).map_err(|e|e.to_string())?;
            DOWNLOAD.fetch_add(4,Ordering::Relaxed);
            if bytes.len()!=4 || u32::from_le_bytes(bytes[..4].try_into().unwrap())!=0 {return Err("RUDA graph index out of bounds".into());}
        }
        Ok(())
    }
    pub fn prepare(&self,client:&ComputeClient<CudaRuntime>,tensors:&[View],n:&NodeSpec,index:usize,inputs:usize,workspace:&mut Vec<View>)
        ->Option<Vec<PreparedKernel<CudaRuntime>>> {
        if (140..=159).contains(&n.op) && tensors[inputs+index].len==0 {return Some(Vec::new());}
        if (146..=159).contains(&n.op) {
            let a=&tensors[n.a as usize];let b=&tensors[n.b as usize];let out=&tensors[inputs+index];
            let dtype=if n.op==147 {b.dtype} else {a.dtype};
            macro_rules! run {($e:ty)=>{{unsafe {
                if n.op==146 {masked_fill::prepare::<$e,CudaRuntime>(client,grid(out.len),RudaDim::new_1d(128),a.arg(),b.arg(),out.arg(),n.scalar,a.dtype==3)}
                else if n.op==147 {where_tensor::prepare::<$e,CudaRuntime>(client,grid(out.len),RudaDim::new_1d(128),a.arg(),b.arg(),tensors[n.scalar as usize].arg(),out.arg())}
                else {compare::prepare::<$e,CudaRuntime>(client,grid(out.len),RudaDim::new_1d(128),a.arg(),b.arg(),out.arg(),n.scalar,n.op)}
            }}};}
            return Some(vec![match dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),3|8=>run!(u8),4=>run!(i64),5=>run!(i32),6=>run!(i16),7=>run!(i8),_=>unreachable!()}]);
        }
        if (140..=145).contains(&n.op) {
            let a=&tensors[n.a as usize];let b=&tensors[n.b as usize];let out=&tensors[inputs+index];
            let code=n.scalar as u32;let axis=if n.op==142 {code as usize} else {(code&7) as usize};
            let start=if n.op==142 {0} else {(code>>3) as usize};
            if n.op==142 && let Some(prepared)=prepare_vector_cat(client,a,b,out,axis) {return Some(vec![prepared]);}
            macro_rules! run {($e:ty)=>{unsafe{transform::prepare::<$e,CudaRuntime>(client,grid(out.len),RudaDim::new_1d(128),a.arg(),b.arg(),out.arg(),n.op,axis,start)}};}
            return Some(vec![match a.dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),3|8=>run!(u8),4=>run!(i64),5=>run!(i32),6=>run!(i16),7=>run!(i8),_=>unreachable!()}]);
        }
        if !(134..=139).contains(&n.op) {return None;}
        let a=&tensors[n.a as usize];let b=&tensors[n.b as usize];let out=&tensors[inputs+index];
        let invalid=self.invalid.as_ref().unwrap();let mut prepared=Vec::new();
        if n.op<=136 {
            if out.len==0 {return Some(prepared);}
            let axis=if n.op==136 {0} else {n.scalar as usize};
            let inner=a.shape[axis+1..].iter().product::<usize>();
            if n.op!=135 && inner>=4 && inner%4==0 && a.contiguous() && out.contiguous()
                && [a,out].iter().all(|v|v.handle.offset_start.unwrap_or(0)%(4*element_bytes(v.dtype)) as u64==0) {
                macro_rules! data {($e:ty)=>{{
                    macro_rules! indices {($i:ty)=>{indexed_vector::prepare::<$e,$i,CudaRuntime>(client,grid(out.len/4),RudaDim::new_1d(128),
                        a.arg(),b.arg(),out.arg(),invalid.arg(),inner,a.shape[axis])};}
                    if b.dtype==4 {indices!(i64)} else {indices!(i32)}
                }};}
                prepared.push(match a.dtype {0=>data!(f32),1=>data!(f16),2=>data!(bf16),3|8=>data!(u8),4=>data!(i64),5=>data!(i32),6=>data!(i16),7=>data!(i8),_=>unreachable!()});
                return Some(prepared);
            }
            macro_rules! data {($e:ty)=>{{
                macro_rules! indices {($i:ty)=>{unsafe{indexed::prepare::<$e,$i,CudaRuntime>(client,grid(out.len),RudaDim::new_1d(128),a.arg(),b.arg(),out.arg(),invalid.arg(),axis,n.op)}};}
                if b.dtype==4 {indices!(i64)} else {indices!(i32)}
            }};}
            prepared.push(match a.dtype {0=>data!(f32),1=>data!(f16),2=>data!(bf16),3|8=>data!(u8),4=>data!(i64),5=>data!(i32),6=>data!(i16),7=>data!(i8),_=>unreachable!()});
        } else {
            let scratch=self.backward[index].as_ref().unwrap();
            if let BackwardBuffers::Segmented=scratch {
                let (sorting,buffers)=super::operation_scatter::prepare_segments(client,b,out.shape[0],workspace);
                prepared.extend(sorting);
                macro_rules! check {($i:ty)=>{bounds_check::prepare::<$i,CudaRuntime>(client,grid(b.len),RudaDim::new_1d(128),b.arg(),invalid.arg(),0i64,out.shape[0] as i64)};}
                prepared.push(if b.dtype==4 {check!(i64)}else {check!(i32)});
                let vector=out.shape[1]%4==0 && [a,out].iter().all(|v|v.handle.offset_start.unwrap_or(0)%(4*element_bytes(v.dtype)) as u64==0);
                let width=if vector {4}else {1};
                macro_rules! run {($e:ty,$v:ty)=>{embedding_segmented_backward::prepare::<$e,$v,CudaRuntime>(client,grid(out.len/width),RudaDim::new_1d(128),
                    a.arg(),buffers[1].arg(),buffers[2].arg(),out.arg(),n.scalar as i64,n.op==138)};}
                macro_rules! data {($e:ty)=>{if vector {run!($e,Const<4>)}else {run!($e,Const<1>)}};}
                prepared.push(match a.dtype {0=>data!(f32),1=>data!(f16),2=>data!(bf16),_=>unreachable!()});
                *workspace=buffers;return Some(prepared);
            }
            let BackwardBuffers::Atomic{accumulation,counts}=scratch else {unreachable!()};
            if out.len!=0 {prepared.push(Self::zero(client,accumulation));}
            let counts=counts.as_ref().unwrap_or(invalid);
            if n.op==138 {
                if counts.len!=0 {prepared.push(Self::zero(client,counts));}
                macro_rules! count {($i:ty)=>{unsafe{frequencies::prepare::<$i,CudaRuntime>(client,grid(b.len),RudaDim::new_1d(128),b.arg(),counts.arg(),invalid.arg())}};}
                if b.len!=0 {prepared.push(if b.dtype==4 {count!(i64)} else {count!(i32)});}
            }
            let axis=if n.op==139 {n.scalar as usize} else {0};let padding=if n.op==139 {-1} else {n.scalar as i64};
            macro_rules! data {($e:ty)=>{{
                macro_rules! indices {($i:ty)=>{unsafe{indexed_backward::prepare::<$e,$i,CudaRuntime>(client,grid(a.len),RudaDim::new_1d(128),a.arg(),b.arg(),accumulation.arg(),counts.arg(),invalid.arg(),padding,axis,n.op)}};}
                if b.dtype==4 {indices!(i64)} else {indices!(i32)}
            }};}
            if a.len!=0 {prepared.push(match a.dtype {0=>data!(f32),1=>data!(f16),2=>data!(bf16),_=>unreachable!()});}
            if out.len!=0 && out.dtype!=0 {
                macro_rules! convert {($e:ty)=>{unsafe{kernels::convert::prepare::<f32,$e,CudaRuntime>(client,grid(out.len),RudaDim::new_1d(128),accumulation.arg(),out.arg())}};}
                prepared.push(if out.dtype==1 {convert!(f16)} else {convert!(bf16)});
            }
        }
        Some(prepared)
    }
}
