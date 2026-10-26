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
    if !matches!(n.op,0|112..=115) {return None;}
    macro_rules! input {($e:ty)=>{{
        if n.op==113 {unsafe{kernels::graph_permute::prepare::<$e,CudaRuntime>(client,grid(out.len),RudaDim::new_1d(128),a.arg(),out.arg(),n.scalar as u32)}}
        else if n.op==115 {unsafe{kernels::graph_expand::prepare::<$e,CudaRuntime>(client,grid(out.len),RudaDim::new_1d(128),a.arg(),out.arg())}}
        else if out.dtype==3 {unsafe{to_bool::prepare::<$e,CudaRuntime>(client,grid(out.len),RudaDim::new_1d(128),a.arg(),out.arg())}}
        else {
            macro_rules! output {($o:ty)=>{unsafe{kernels::convert::prepare::<$e,$o,CudaRuntime>(client,grid(out.len),RudaDim::new_1d(128),a.arg(),out.arg())}};}
            match out.dtype {0=>output!(f32),1=>output!(f16),2=>output!(bf16),4=>output!(i64),5=>output!(i32),6=>output!(i16),7=>output!(i8),8=>output!(u8),_=>unreachable!()}
        }
    }};}
    Some(match a.dtype {0=>input!(f32),1=>input!(f16),2=>input!(bf16),3|8=>input!(u8),4=>input!(i64),5=>input!(i32),6=>input!(i16),7=>input!(i8),_=>unreachable!()})
}

struct BackwardBuffers {accumulation:View,counts:Option<View>}
pub(super) struct IndexBuffers {invalid:Option<View>,backward:Vec<Option<BackwardBuffers>>}
impl IndexBuffers {
    pub fn enabled(&self)->bool {self.invalid.is_some()}
    pub fn new(client:&ComputeClient<CudaRuntime>,tensors:&[View],nodes:&[NodeSpec],inputs:usize)->Self {
        let invalid=nodes.iter().any(|n|(134..=139).contains(&n.op)).then(||View::packed(client.empty(4),vec![1],5));
        let backward=nodes.iter().enumerate().map(|(index,node)| {
            if !(137..=139).contains(&node.op) {return None;}
            let out=&tensors[inputs+index];
            let accumulation=View::packed(if out.dtype==0 {out.handle.clone()} else {client.empty(out.len.checked_mul(4).expect("gradient workspace overflow").max(4))},out.shape.clone(),0);
            let counts=(node.op==138).then(||View::packed(client.empty(out.shape[0].checked_mul(4).expect("frequency workspace overflow").max(4)),vec![out.shape[0]],5));
            Some(BackwardBuffers{accumulation,counts})
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
    pub fn prepare(&self,client:&ComputeClient<CudaRuntime>,tensors:&[View],n:&NodeSpec,index:usize,inputs:usize)
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
            macro_rules! data {($e:ty)=>{{
                macro_rules! indices {($i:ty)=>{unsafe{indexed::prepare::<$e,$i,CudaRuntime>(client,grid(out.len),RudaDim::new_1d(128),a.arg(),b.arg(),out.arg(),invalid.arg(),axis,n.op)}};}
                if b.dtype==4 {indices!(i64)} else {indices!(i32)}
            }};}
            prepared.push(match a.dtype {0=>data!(f32),1=>data!(f16),2=>data!(bf16),3|8=>data!(u8),4=>data!(i64),5=>data!(i32),6=>data!(i16),7=>data!(i8),_=>unreachable!()});
        } else {
            let scratch=self.backward[index].as_ref().unwrap();
            if out.len!=0 {prepared.push(Self::zero(client,&scratch.accumulation));}
            let counts=scratch.counts.as_ref().unwrap_or(invalid);
            if n.op==138 {
                if counts.len!=0 {prepared.push(Self::zero(client,counts));}
                macro_rules! count {($i:ty)=>{unsafe{frequencies::prepare::<$i,CudaRuntime>(client,grid(b.len),RudaDim::new_1d(128),b.arg(),counts.arg(),invalid.arg())}};}
                if b.len!=0 {prepared.push(if b.dtype==4 {count!(i64)} else {count!(i32)});}
            }
            let axis=if n.op==139 {n.scalar as usize} else {0};let padding=if n.op==139 {-1} else {n.scalar as i64};
            macro_rules! data {($e:ty)=>{{
                macro_rules! indices {($i:ty)=>{unsafe{indexed_backward::prepare::<$e,$i,CudaRuntime>(client,grid(a.len),RudaDim::new_1d(128),a.arg(),b.arg(),scratch.accumulation.arg(),counts.arg(),invalid.arg(),padding,axis,n.op)}};}
                if b.dtype==4 {indices!(i64)} else {indices!(i32)}
            }};}
            if a.len!=0 {prepared.push(match a.dtype {0=>data!(f32),1=>data!(f16),2=>data!(bf16),_=>unreachable!()});}
            if out.len!=0 && out.dtype!=0 {
                macro_rules! convert {($e:ty)=>{unsafe{kernels::convert::prepare::<f32,$e,CudaRuntime>(client,grid(out.len),RudaDim::new_1d(128),scratch.accumulation.arg(),out.arg())}};}
                prepared.push(if out.dtype==1 {convert!(f16)} else {convert!(bf16)});
            }
        }
        Some(prepared)
    }
}
