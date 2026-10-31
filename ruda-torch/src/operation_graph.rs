use super::*;
use ruda_driver_cuda::graph::CudaGraph;
use ruda_kernel::{tensor::RudaTensor,quantization::{float8,packed::{PackedTensor,PackedFormat}}};
use ruda_core::tensor::Shape;
use rudnn::{dense_attention as dense,varlen_attention as varlen};
use std::sync::Mutex;

pub struct NativeOperationGraph {stream:u64,state:Mutex<State>}
struct State {
    graphs:Vec<CudaGraph>,client:ComputeClient<CudaRuntime>,tensors:Vec<View>,scratch:Vec<View>,
    _sequence_workspace:Vec<RudaTensor<CudaRuntime>>,
    kind:u32,integers:Vec<i64>,scalars:Vec<f32>,kernels:usize,attention_node:Option<usize>,
}

pub(super) fn packed(v:&View) {
    assert!(v.len<=u32::MAX as usize);
    if v.len==0 {return;}
    let mut stride=1usize;
    for (&size,&actual) in v.shape.iter().zip(&v.strides).rev() {
        assert!(size<=1 || actual==stride,"operation graph requires contiguous storage");
        stride=stride.checked_mul(size).expect("operation graph shape overflow");
    }
}
fn flag(x:i64)->bool {assert!(x==0 || x==1);x!=0}
fn tensor(c:&ComputeClient<CudaRuntime>,v:&View)->RudaTensor<CudaRuntime> {
    let mut t=primitives::tensor(v);t.client=c.clone();t
}
fn grid(n:usize)->RudaCount {RudaCount::Static(u32::try_from(n.div_ceil(128)).expect("operation graph grid overflow"),1,1)}
fn same(a:&View,b:&View) {assert_eq!(a.shape,b.shape);assert_eq!(a.dtype,b.dtype);}

fn validate(kind:u32,vs:&[View],ints:&[i64],floats:&[f32]) {
    assert!(kind<=44);
    if matches!(kind,43|44) {assert!(floats.is_empty());indexed_experts::float8_block::validate(vs,ints,kind==44);return;}
    if matches!(kind,41|42) {assert!(floats.is_empty());indexed_experts::float8::validate(vs,ints,kind==41);return;}
    if kind==40 {assert!(floats.is_empty());indexed_experts::quantized::validate(vs,ints);return;}
    if kind==39 {assert!(floats.is_empty());indexed_experts::validate_activity(vs,ints);return;}
    if kind==38 {assert!(floats.is_empty());assert_eq!(ints.len(),4);indexed_experts::validate(vs,true,ints);return;}
    if kind==37 {
        assert_eq!(floats.len(),1);dense_attention::validate_dropout(vs,ints,floats[0]);return;
    }
    if matches!(kind,35|36) {
        assert_eq!(ints.len(),5);validate(kind-35,vs,&ints[..3],floats);
        let q=&vs[0];let h=q.shape[1] as i64;
        assert!(ints[3]>=h && ints[4]>=0 && ints[4]<=ints[3]-h);
        (q.shape[0] as i64).checked_mul(ints[3]).and_then(|n|n.checked_mul(q.shape[2] as i64))
            .and_then(|n|n.checked_mul(vs[1].shape[2] as i64)).expect("global attention RNG index overflow");
        return;
    }
    if matches!(kind,33|34) {
        assert!(floats.is_empty());float8_shard::validate(vs,kind==34,ints);return;
    }
    if matches!(kind,31|32) {
        assert!(floats.is_empty());indexed_experts::validate(vs,kind==32,ints);return;
    }
    if kind==30 {
        assert_eq!((vs.len(),ints.len(),floats.len()),(8,3,3));assert!(ints[0]>0);flag(ints[1]);flag(ints[2]);
        assert!(floats.iter().all(|&n|n.is_finite() && n>0.0));
        for v in vs {packed(v);}
        for i in [0,1,2] {assert_eq!(vs[i].dtype,0);same(&vs[i],&vs[i+4]);}
        assert_eq!(vs[0].shape,vec![3]);assert_eq!(vs[2].shape,vec![3,3]);
        assert_eq!(vs[1].shape.len(),3);assert_eq!(vs[1].shape[0],3);assert_eq!(vs[1].shape[2],2);assert!(vs[1].shape[1]>0);
        assert_eq!(vs[3].dtype,4);assert!(vs[3].shape.is_empty());same(&vs[3],&vs[7]);return;
    }
    if kind==29 {
        assert_eq!((vs.len(),ints.len(),floats.len()),(6,1,0));assert!(matches!(ints[0],12|13));
        for v in vs {packed(v);}
        assert!(vs[0].dtype<=2);assert_eq!(vs[3].dtype,ints[0] as u32);assert_eq!(vs[3].shape,vs[0].shape);
        for i in [1,2,4,5] {assert_eq!(vs[i].dtype,0);}
        assert!(vs[1].shape.is_empty());same(&vs[1],&vs[4]);
        assert_eq!(vs[2].shape,vec![3]);same(&vs[2],&vs[5]);return;
    }
    if kind>=27 {
        assert!(floats.is_empty());
        assert_eq!((vs.len(),ints.len()),if kind==27 {(3,5)} else {(5,4)});
        for v in vs {packed(v);assert!(v.shape.iter().all(|&d|d<=u32::MAX as usize));}
        assert!(ints.iter().all(|&i|i>=0 && i<=u32::MAX as i64));
        if kind==27 {
            assert!(vs[0].dtype<=2 && matches!(ints[0],12|13));
            assert_eq!(vs[0].shape.len(),2);assert_eq!(vs[0].shape,vs[1].shape);assert_eq!(vs[1].dtype,ints[0] as u32);
            let (rows,cols)=(ints[1] as usize,ints[2] as usize);
            assert!(rows>0 && cols>0 && cols%32==0 && rows.checked_mul(cols).is_some_and(|n|n<=u32::MAX as usize-32));
            assert!(ints[3]<=16);flag(ints[4]);
            assert_eq!(vs[2].dtype,0);assert_eq!(vs[2].shape,vec![vs[0].shape[0].div_ceil(rows),vs[0].shape[1].div_ceil(cols)]);
        } else {
            assert!(matches!(vs[0].dtype,12|13) && matches!(vs[1].dtype,12|13));
            assert!(ints[0]>0 && ints[1]>0 && ints[2]>0 && ints[2]%32==0 && ints[3]<=2);
        }
        return;
    }
    if kind==26 {operation_sort::validate(vs,ints,floats);return;}
    if kind==25 {operation_scatter::validate(vs,ints,floats);return;}
    if kind>=23 {
        assert_eq!((vs.len(),ints.len(),floats.len()),if kind==23 {(4,1,0)}else {(6,0,0)});
        if kind==23 {assert!((0..=2).contains(&ints[0]));}
        for v in vs {packed(v);assert!(v.shape.iter().all(|&d|d<=u32::MAX as usize));}
        let (x,w,ids)=(&vs[0],&vs[1],&vs[2]);
        assert_eq!(x.shape.len(),2);assert_eq!(w.shape.len(),3);assert_eq!(ids.shape,vec![x.shape[0]]);
        assert!(x.dtype<=2 && w.dtype==x.dtype && matches!(ids.dtype,4|5));
        assert!(x.shape[1]>0 && w.shape[1]>0 && (w.shape[0]>0 || x.shape[0]==0));assert_eq!(x.shape[1],w.shape[2]);
        assert_eq!(vs[3].shape,vec![x.shape[0],w.shape[1]]);assert_eq!(vs[3].dtype,x.dtype);
        if kind==24 {same(x,&vs[4]);same(w,&vs[5]);}
        return;
    }
    if kind>=21 {
        assert_eq!((vs.len(),ints.len(),floats.len()),(if kind==21 {3}else {4},2,1));
        flag(ints[0]);flag(ints[1]);assert!(floats[0].is_finite() && floats[0]>0.0);
        for v in vs {packed(v);}
        assert!(vs[0].dtype<=2 && matches!(vs[1].dtype,4|5));
        assert_eq!(vs[0].shape.len(),2);assert_eq!(vs[1].shape.len(),2);
        assert_eq!(vs[0].shape[0],vs[1].shape[0]);assert!(vs[0].shape[1]>0 && (1..=vs[0].shape[1].min(64)).contains(&vs[1].shape[1]));
        assert_eq!(vs[2].shape,vs[1].shape);assert_eq!(vs[2].dtype,0);
        if kind==22 {same(&vs[0],&vs[3]);}
        return;
    }
    if kind==20 {
        for v in vs {packed(v);}
        operation_reduce::validate(vs,ints,floats);return;
    }
    if kind==19 {
        for v in vs {packed(v);}
        operation_transform::validate(vs,ints,floats);return;
    }
    let (count,ni,nf)=match kind {0=>(6,3,2),1=>(10,3,2),2=>(6,4,3),3=>(9,4,3),4=>(5,3,0),5=>(3,0,0),6=>(6,6,0),7=>(5,4,0),8=>(7,5,0),9=>(3,3,0),10=>(9,2,1),11=>(15,1,1),12=>(8,1,1),13=>(3,2,0),14|15=>(2,1,0),16=>(vs.len(),1,0),17=>(vs.len(),5,0),18=>(vs.len(),1,2),_=>unreachable!()};
    assert_eq!((vs.len(),ints.len(),floats.len()),(count,ni,nf));
    for (i,v) in vs.iter().enumerate() {if !(kind<=1 && i==3) {packed(v);}else {assert!(v.len<=u32::MAX as usize);}}
    if kind>=17 {operation_pointwise::validate(vs,kind,ints);}
    else if kind<=3 {
        let (q,k,v)=(&vs[0],&vs[1],&vs[2]);assert!(q.dtype<=2);assert_eq!(k.dtype,q.dtype);assert_eq!(v.dtype,q.dtype);
        assert!(floats[0].is_finite() && (0.0..=1.0).contains(&floats[1]));flag(ints[1]);
        let (rows,shape)=if kind<=1 {
            assert!((0..=2).contains(&ints[0]));flag(ints[2]);assert!(ints[1]==0 || ints[0]==0);
            assert!(ints[2]==0 || (kind==1 && ints[0]==2));
            for x in [q,k,v] {assert_eq!(x.shape.len(),4);}
            let (b,h,l,d)=(q.shape[0],q.shape[1],q.shape[2],q.shape[3]);
            let (kh,s,dv)=(k.shape[1],k.shape[2],v.shape[3]);
            assert!(h>0 && kh>0 && h%kh==0 && d>0 && dv>0);
            assert_eq!(k.shape,vec![b,kh,s,d]);assert_eq!(v.shape,vec![b,kh,s,dv]);
            let rows=b.checked_mul(h).and_then(|x|x.checked_mul(l)).expect("attention rows overflow");
            let scores=rows.checked_mul(s).filter(|n|*n<=u32::MAX as usize).expect("attention scores overflow");
            assert_eq!(vs[3].dtype,0);
            if ints[0]!=0 {assert_eq!(vs[3].len,scores);}
            if kind==1 {
                assert_eq!(vs[9].dtype,0);
                if ints[2]!=0 {
                    let span=if vs[3].len==0 {0} else {1+vs[3].shape.iter().zip(&vs[3].strides).map(|(d,s)|(d-1)*s).sum::<usize>()};
                    assert_eq!(vs[9].len,span);
                }
            }
            (rows,vec![b,h,l,dv])
        } else {
            flag(ints[0]);
            assert!(ints[2]>=-1 && ints[3]>=-1 && floats[2].is_finite() && floats[2]>=0.0);
            for x in [q,k,v] {assert_eq!(x.shape.len(),3);}
            let (tokens,h,d)=(q.shape[0],q.shape[1],q.shape[2]);let (keys,kh,dv)=(k.shape[0],k.shape[1],v.shape[2]);
            assert!(h>0 && kh>0 && h%kh==0 && d>0 && dv>0);
            assert_eq!(k.shape,vec![keys,kh,d]);assert_eq!(v.shape,vec![keys,kh,dv]);
            assert_eq!(vs[3].dtype,4);assert_eq!(vs[3].shape.len(),2);assert_eq!(vs[3].shape[1],6);assert!(vs[3].shape[0]>0);
            (tokens.checked_mul(h).expect("varlen rows overflow"),vec![tokens,h,dv])
        };
        assert!(rows>0 && rows<=u32::MAX as usize,"empty-query attention graphs are not supported");
        assert_eq!(vs[4].shape,shape);assert_eq!(vs[4].dtype,q.dtype);assert_eq!(vs[5].dtype,0);assert_eq!(vs[5].shape,vec![rows,2]);
        if kind%2==1 {same(q,&vs[6]);same(k,&vs[7]);same(v,&vs[8]);}
    } else if kind==4 {
        flag(ints[0]);flag(ints[1]);assert!((0..=2).contains(&ints[2]));
        assert!(matches!(vs[0].dtype,12|13) && matches!(vs[1].dtype,12|13));
    } else if kind==5 {
        assert!(vs[0].dtype<=2 && matches!(vs[2].dtype,12|13));
        assert_eq!(vs[1].dtype,0);assert_eq!(vs[1].len,1);assert_eq!(vs[0].shape,vs[2].shape);assert!(vs[2].len>0);
    } else if kind<=7 {
        for &word in &ints[..4] {assert!(word>=0 && word<=u32::MAX as i64);}
        assert!(ints[1]>0 && ints[2]>0);
        if kind==6 {flag(ints[4]);flag(ints[5]);}
    } else if kind==8 {
        assert!(matches!(ints[0],12|13) && matches!(ints[1],12|13) && (0..=2).contains(&ints[2]) && (0..=16).contains(&ints[3]));flag(ints[4]);
        assert!(vs[0].dtype<=2 && vs[1].dtype<=2 && vs[2].dtype<=2 && vs[2].len>0);
        assert_eq!(vs[0].shape.len(),2);assert_eq!(vs[1].shape.len(),2);assert_eq!(vs[0].shape[1],vs[1].shape[1]);
        assert_eq!(vs[2].shape,vec![vs[0].shape[0],vs[1].shape[0]]);
        for (i,j,format) in [(0,3,ints[0]),(1,4,ints[1])] {assert_eq!(vs[i].shape,vs[j].shape);assert_eq!(vs[j].dtype,format as u32);}
        for index in [5,6] {assert_eq!(vs[index].dtype,0);assert_eq!(vs[index].len,1);}
    } else if kind==9 {
        assert!(matches!(ints[0],12|13) && (0..=16).contains(&ints[1]));flag(ints[2]);
        assert!(vs[0].dtype<=2);assert_eq!(vs[1].dtype,ints[0] as u32);assert_eq!(vs[0].shape,vs[1].shape);
        assert_eq!(vs[2].dtype,0);assert_eq!(vs[2].len,1);
    } else if kind<=12 {
        assert!(ints[0]>0 && ints[0]<=u32::MAX as i64 && floats[0].is_finite());
        assert_eq!(vs[0].shape.len(),4); assert!(vs[0].shape[0]>0);
        if kind==10 {
            flag(ints[1]); assert_eq!(vs[5].shape.len(),4);
            let mut shape=vec![if ints[1]!=0 {vs[0].shape[2].div_ceil(ints[0] as usize)} else {0}];
            shape.extend(&vs[5].shape); assert_eq!(vs[8].shape,shape); assert_eq!(vs[8].dtype,0);
        }
        if kind==12 { assert!(vs.iter().all(|v|v.dtype==0)); }
    } else if kind==13 {
        flag(ints[0]);flag(ints[1]);
    } else if kind==16 {
        assert!(vs.len()>=2 && ints[0]>=0 && (ints[0] as usize)<vs.last().unwrap().shape.len());
        assert!(vs[0].dtype<=9 || matches!(vs[0].dtype,12|13));
    } else {
        assert_eq!(vs[0].shape,vs[1].shape);
        if kind==14 {
            assert!(vs[0].dtype<=9 && vs[1].dtype<=9 && vs[1].dtype!=3);
            assert!(ints[0]>=0 && (ints[0] as usize)<vs[0].shape.len());
        } else {
            same(&vs[0],&vs[1]);assert!(vs[0].dtype<=9 || matches!(vs[0].dtype,12|13));
            assert!(vs[0].shape.len()<=32 && ints[0]>=0 && (ints[0] as u64)<(1u64<<vs[0].shape.len()));
        }
    }
}

fn attention(c:&ComputeClient<CudaRuntime>,vs:&[View],scratch:&[View],kind:u32,ints:&[i64],floats:&[f32],seed:u64,counter:u64)->PreparedKernel<CudaRuntime> {
    if kind==37 {return dense_attention::prepare_dropout(c,vs,ints,floats[0],seed,counter);}
    let kind=if matches!(kind,35|36) {kind-35} else {kind};
    let ts:Vec<_>=vs.iter().map(|v|tensor(c,v)).collect();let q=&vs[0];
    let rows=vs[5].shape[0];let blocks=RudaCount::Static(rows.div_ceil(4) as u32,1,1);let dim=RudaDim::new_1d(128);
    if kind<=1 {
        if kind==0 && floats[1]==0.0 {
            if let Some(prepared)=super::attention_prefill::prepare(c,&vs[0],&vs[1],&vs[2],&vs[4],&vs[3],&vs[5],
                ints[0] as u32,ints[1]!=0,floats[0]) {return prepared;}
        }
        let (h,l,d)=(q.shape[1],q.shape[2],q.shape[3]);let (kh,s,dv)=(vs[1].shape[1],vs[1].shape[2],vs[2].shape[3]);
        let (global_heads,head_start)=if ints.len()==5 {(ints[3],ints[4])}else {(h as i64,0)};
        let scores=(q.shape[0] as i64).checked_mul(global_heads).and_then(|n|n.checked_mul(l as i64))
            .and_then(|n|n.checked_mul(s as i64)).expect("global attention RNG index overflow");
        counter.checked_add((scores as u64).div_ceil(4)).expect("attention RNG counter overflow");
        unsafe {if kind==0 {dense::forward::prepare::<CudaRuntime>(c,blocks,dim,
            ts[0].clone().into_array_arg(),ts[1].clone().into_array_arg(),ts[2].clone().into_array_arg(),ts[3].clone().into_tensor_arg(),
            ts[4].clone().into_array_arg(),ts[5].clone().into_array_arg(),h as u32,kh as u32,l as u32,s as u32,floats[0],floats[1],
            seed as u32,(seed>>32) as u32,counter as u32,(counter>>32) as u32,global_heads,head_start,
            d,dv,32,ints[1]!=0,ints[0] as u32,floats[1]!=0.0,ts[0].dtype.into())}
        else {dense::backward::prepare::<CudaRuntime>(c,blocks,dim,
            ts[0].clone().into_array_arg(),ts[1].clone().into_array_arg(),ts[2].clone().into_array_arg(),ts[3].clone().into_tensor_arg(),
            ts[4].clone().into_array_arg(),ts[5].clone().into_array_arg(),ts[6].clone().into_array_arg(),tensor(c,&scratch[0]).into_array_arg(),
            tensor(c,&scratch[1]).into_array_arg(),ts[9].clone().into_array_arg(),h as u32,kh as u32,l as u32,s as u32,floats[0],floats[1],
            seed as u32,(seed>>32) as u32,counter as u32,(counter>>32) as u32,global_heads,head_start,
            d,dv,32,ints[1]!=0,ints[0] as u32,floats[1]!=0.0,ints[2]!=0,ts[0].dtype.into())}}
    } else {
        let (h,d,kh,dv)=(q.shape[1],q.shape[2],vs[1].shape[1],vs[2].shape[2]);let docs=vs[3].shape[0] as u32;
        unsafe {if kind==2 {varlen::forward::prepare::<CudaRuntime>(c,blocks,dim,
            ts[0].clone().into_array_arg(),ts[1].clone().into_array_arg(),ts[2].clone().into_array_arg(),ts[3].clone().into_array_arg(),
            ts[4].clone().into_array_arg(),ts[5].clone().into_array_arg(),h as u32,kh as u32,docs,floats[0],floats[1],floats[2],ints[2],ints[3],
            seed as u32,(seed>>32) as u32,counter as u32,(counter>>32) as u32,d,dv,32,ints[0]!=0,ints[1]!=0,floats[1]!=0.0,floats[2]!=0.0,ts[0].dtype.into())}
        else {varlen::backward::prepare::<CudaRuntime>(c,blocks,dim,
            ts[0].clone().into_array_arg(),ts[1].clone().into_array_arg(),ts[2].clone().into_array_arg(),ts[3].clone().into_array_arg(),
            ts[4].clone().into_array_arg(),ts[5].clone().into_array_arg(),ts[6].clone().into_array_arg(),tensor(c,&scratch[0]).into_array_arg(),tensor(c,&scratch[1]).into_array_arg(),
            h as u32,kh as u32,docs,floats[0],floats[1],floats[2],ints[2],ints[3],seed as u32,(seed>>32) as u32,counter as u32,(counter>>32) as u32,
            d,dv,32,ints[0]!=0,ints[1]!=0,floats[1]!=0.0,floats[2]!=0.0,ts[0].dtype.into())}}
    }
}

fn prepare(c:&ComputeClient<CudaRuntime>,vs:&[View],scratch:&[View],kind:u32,ints:&[i64],floats:&[f32])->(Vec<PreparedKernel<CudaRuntime>>,Option<usize>) {
    let kind=if matches!(kind,35|36) {kind-35} else {kind};
    let mut nodes=Vec::new();let mut update=None;
    if kind<=3 {
        if kind%2==1 {
            for v in scratch.iter().chain(if kind==1 {Some(&vs[9])} else {None}) {
                if v.len>0 {nodes.push(unsafe {dense::zero::prepare::<CudaRuntime>(c,grid(v.len),RudaDim::new_1d(128),tensor(c,v).into_array_arg())});}
            }
        }
        update=Some(nodes.len());nodes.push(attention(c,vs,scratch,kind,ints,floats,0,0));
        if kind%2==1 {
            for (src,dst) in scratch.iter().zip([&vs[7],&vs[8]]) {
                if dst.len>0 {nodes.push(unsafe {dense::narrow::prepare::<CudaRuntime>(c,grid(dst.len),RudaDim::new_1d(128),
                    tensor(c,src).into_array_arg(),tensor(c,dst).into_array_arg(),tensor(c,dst).dtype.into())});}
            }
        }
    } else if kind<=7 {
        let ts:Vec<_>=vs.iter().map(|v|tensor(c,v)).collect();
        nodes.push(if kind==4 {
            float8::prepare_scaled_mm(&ts[0],&ts[1],&ts[2],&ts[3],&ts[4],if vs[0].dtype==12 {18} else {19},if vs[1].dtype==12 {18} else {19},flag(ints[0]),flag(ints[1]),ints[2] as u32)
        } else if kind==5 {unsafe {float8::quantize::prepare::<CudaRuntime>(c,grid(vs[2].len),RudaDim::new_1d(128),ts[0].clone().into_array_arg(),ts[1].clone().into_array_arg(),
            ts[2].clone().into_array_arg(),if vs[2].dtype==12 {18} else {19},ts[0].dtype.into())}}
        else {
            let base=if kind==6 {1} else {0};
            let weight=PackedTensor::new(ts[base].clone(),ts[base+1].clone(),ts[base+2].clone(),ts[base+3].clone(),Shape::new([ints[3] as usize,ints[2] as usize]),PackedFormat::from_code(ints[0] as u32),ints[1] as usize);
            if kind==6 {weight.prepare_linear(&ts[0],&ts[5],flag(ints[4]),flag(ints[5]))} else {weight.prepare_decode(&ts[4])}
        });
    } else {
        let ts:Vec<_>=vs.iter().map(|v|tensor(c,v)).collect();
        if kind==8 {
            for (input,encoded,scale,scratch_index,format) in [(0,3,5,0,ints[0]),(1,4,6,1,ints[1])] {
                nodes.extend(float8::prepare_current_quantize(&ts[input],&ts[encoded],&ts[scale],&tensor(c,&scratch[scratch_index]),
                    if format==12 {18} else {19},ints[3] as u32,flag(ints[4])));
            }
            nodes.push(float8::prepare_scaled_mm(&ts[3],&ts[4],&ts[5],&ts[6],&tensor(c,&scratch[2]),
                if ints[0]==12 {18} else {19},if ints[1]==12 {18} else {19},false,true,ints[2] as u32));
            if vs[2].dtype!=0 {
                macro_rules! cast {($e:ty)=>{unsafe {kernels::convert::prepare::<f32,$e,CudaRuntime>(c,grid(vs[2].len),RudaDim::new_1d(128),scratch[2].arg(),vs[2].arg())}};}
                nodes.push(if vs[2].dtype==1 {cast!(f16)} else {cast!(bf16)});
            }
        } else {nodes.extend(float8::prepare_current_quantize(&ts[0],&ts[1],&ts[2],&tensor(c,&scratch[0]),
            if ints[0]==12 {18} else {19},ints[1] as u32,flag(ints[2])));}
    }
    (nodes,update)
}

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_operation_graph_api_version()->u32 {1}

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_delayed_float8_graph_api_version()->u32 {1}

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_sequence_graph_api_version()->u32 {1}

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_pointwise_graph_api_version()->u32 {1}

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_transform_graph_api_version()->u32 {1}

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_reduction_graph_api_version()->u32 {1}

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_router_graph_api_version()->u32 {1}

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_grouped_graph_api_version()->u32 {1}

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_scatter_graph_api_version()->u32 {1}

fn sequence(c:&ComputeClient<CudaRuntime>,vs:&[View],kind:u32,ints:&[i64],floats:&[f32])
    ->rudnn::sequence_graph::SequenceKernelPlan<CudaRuntime>
{
    use rudnn::{kda,gated_delta as delta,sequence_graph::SequenceKernelPlan};
    if kind==13 { return SequenceKernelPlan {kernels:sequence_training::prepare_triangular(c,&vs[0],&vs[1],&vs[2],flag(ints[0]),flag(ints[1])),workspace:Vec::new()}; }
    let ts:Vec<_>=vs.iter().map(|v|tensor(c,v)).collect();
    if kind==12 {
        let input=delta::GatedDeltaInput {query:ts[0].clone(),key:ts[1].clone(),value:ts[2].clone(),
            beta:ts[3].clone(),log_decay:ts[4].clone(),initial_state:ts[5].clone(),query_scale:floats[0]};
        let out=delta::GatedDeltaOutput {output:ts[6].clone(),final_state:ts[7].clone()};
        delta::prepare_chunk_gated_delta_rule(&input,&out,ints[0] as usize).expect("gated delta graph preparation failed")
    } else {
        let input=kda::KdaInput {query:ts[0].clone(),key:ts[1].clone(),value:ts[2].clone(),
            beta:ts[3].clone(),log_decay:ts[4].clone(),initial_state:ts[5].clone(),query_scale:floats[0]};
        if kind==10 {
            let out=kda::KdaOutput {output:ts[6].clone(),final_state:ts[7].clone(),boundaries:flag(ints[1]).then(||ts[8].clone())};
            kda::prepare_chunk_kda(&input,&out,ints[0] as usize).expect("KDA forward graph preparation failed")
        } else {
            let out=kda::KdaGradients {query:ts[9].clone(),key:ts[10].clone(),value:ts[11].clone(),
                beta:ts[12].clone(),log_decay:ts[13].clone(),initial_state:ts[14].clone()};
            kda::prepare_chunk_kda_backward(&input,&ts[6],&ts[7],&ts[8],&out,ints[0] as usize)
                .expect("KDA backward graph preparation failed")
        }
    }
}

/// 0 build, 1 replay with RNG scalar update, 2 wait, 3 query, 4 close.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_operation_graph(command:u32,plan:*mut *mut NativeOperationGraph,kind:u32,
    descriptors:*const Descriptor,count:usize,integers:*const i64,integer_count:usize,scalars:*const f32,scalar_count:usize,
    seed:u64,counter:u64,result:*mut u64)->i32 {
    checked(|| {
        assert!(!plan.is_null() && !result.is_null());unsafe {*result=0;}
        if command==0 {
            assert!(unsafe{(*plan).is_null()});assert!((count<=16 || matches!(kind,31|32|38) && count<=36
                || kind==40 && count<=67 || matches!(kind,41|42) && count<=36 || matches!(kind,43|44) && count<=37
                || kind==16 && count<=u32::MAX as usize)
                && integer_count<=(if kind==40 {99}else if kind==19 {19}else if matches!(kind,43|44) {15}else {8}) && scalar_count<=4);
            let vs:Vec<_>=unsafe{abi::slice(descriptors,count)}.iter().map(|d|unsafe{abi::view(d)}).collect();
            let ints=unsafe{abi::slice(integers,integer_count)}.to_vec();let floats=unsafe{abi::slice(scalars,scalar_count)}.to_vec();
            validate(kind,&vs,&ints,&floats);let client=client().fixed_execution_queue();
            let mut scratch=if matches!(kind,1|3|36) {vs[1..3].iter().map(|v|View::packed(client.empty(v.len.max(1).checked_mul(4).expect("attention workspace overflow")),v.shape.clone(),0)).collect()}else {Vec::new()};
            if matches!(kind,8|9|29) {
                for value in &vs[..if kind==8 {2} else {1}] {
                    let blocks=value.len.div_ceil(4096).max(1);
                    scratch.push(View::packed(client.empty(blocks*8),vec![blocks,2],0));
                }
                if kind==8 {scratch.push(View::packed(if vs[2].dtype==0 {vs[2].handle.clone()}else {client.empty(vs[2].len.checked_mul(4).expect("FP8 graph accumulator overflow"))},vs[2].shape.clone(),0));}
            }
            let (prepared,attention_node,workspace)=if matches!(kind,43|44) {
                let plan=indexed_experts::float8_block::prepare(&client,&vs,&ints,kind==44);(plan.kernels,None,plan.workspace)
            } else if matches!(kind,41|42) {
                let plan=if kind==41 {indexed_experts::float8::prepare_statistics(&client,&vs,&ints)}
                    else {indexed_experts::float8::prepare_product(&client,&vs,&ints)};
                (plan.kernels,None,plan.workspace)
            } else if kind==40 {
                let plan=indexed_experts::quantized::prepare(&client,&vs,&ints);(plan.kernels,None,plan.workspace)
            } else if kind==39 {
                (vec![indexed_experts::prepare_activity(&client,&vs,&ints)],None,Vec::new())
            } else if kind==38 {
                let plan=indexed_experts::prepare(&client,&vs,true,&ints);(plan.kernels,None,plan.workspace)
            } else if kind==37 {
                (vec![attention(&client,&vs,&scratch,kind,&ints,&floats,0,0)],Some(0),Vec::new())
            } else if matches!(kind,35|36) {
                let (nodes,update)=prepare(&client,&vs,&scratch,kind,&ints,&floats);(nodes,update,Vec::new())
            } else if matches!(kind,33|34) {
                (float8_shard::prepare(&client,&vs,kind==34,&ints),None,Vec::new())
            } else if matches!(kind,31|32) {
                let plan=indexed_experts::prepare(&client,&vs,kind==32,&ints);(plan.kernels,None,plan.workspace)
            } else if kind==30 {
                let ts:Vec<_>=vs.iter().map(|v|tensor(&client,v)).collect();
                (vec![unsafe {float8::delayed_advance::prepare::<CudaRuntime>(&client,RudaCount::Static(3,1,1),RudaDim::new_1d(32),
                    ts[0].clone().into_array_arg(),ts[1].clone().into_array_arg(),ts[2].clone().into_array_arg(),ts[3].clone().into_array_arg(),
                    ts[4].clone().into_array_arg(),ts[5].clone().into_array_arg(),ts[6].clone().into_array_arg(),ts[7].clone().into_array_arg(),
                    vs[1].shape[1] as u32,ints[0],floats[0],floats[1],floats[2],flag(ints[1]),flag(ints[2]))}],None,Vec::new())
            } else if kind==29 {
                let ts:Vec<_>=vs.iter().map(|v|tensor(&client,v)).collect();
                (float8::prepare_delayed_quantize(&ts[0],&ts[1],&ts[2],&ts[3],&ts[4],&ts[5],
                    &tensor(&client,&scratch[0]),if ints[0]==12 {18} else {19}),None,Vec::new())
            } else if kind>=27 {
                use ruda_kernel::quantization::float8_block as block;
                let ts:Vec<_>=vs.iter().map(|v|tensor(&client,v)).collect();
                let format=|dtype|if dtype==12 {18u32} else {19u32};
                let nodes=if kind==28 {
                    block::prepare_scaled_mm(&ts[0],&ts[1],&ts[2],&ts[3],&ts[4],ints[0] as usize,ints[1] as usize,ints[2] as usize,
                        format(vs[0].dtype),format(vs[1].dtype),ints[3] as u32)
                } else if vs[0].len==0 {Vec::new()} else {
                    assert!(vs[2].len<=client.properties().hardware.max_ruda_count.0 as usize);
                    let maximum=if ints[0]==12 {448.0f32} else {57344.0f32};
                    vec![unsafe {block::quantize_blocks::prepare::<CudaRuntime>(&client,RudaCount::Static(vs[2].len as u32,1,1),RudaDim::new_1d(32),
                        ts[0].clone().into_array_arg(),ts[1].clone().into_array_arg(),ts[2].clone().into_array_arg(),
                        vs[0].shape[0] as u32,vs[0].shape[1] as u32,vs[2].shape[1] as u32,ints[1] as u32,ints[2] as u32,
                        maximum*2.0f32.powi(-(ints[3] as i32)),format(ints[0] as u32),flag(ints[4]),ts[0].dtype.into())}]
                };
                (nodes,None,Vec::new())
            } else if kind==26 {
                let (nodes,extra)=operation_sort::prepare(&client,&vs,&ints);scratch.extend(extra);
                (nodes,None,Vec::new())
            } else if kind==25 {
                let (nodes,extra)=operation_scatter::prepare(&client,&vs,&ints);scratch.extend(extra);
                (nodes,None,Vec::new())
            } else if kind>=23 {
                let plan=blas_bridge::prepare_grouped(&client,&vs,kind==24,&ints);(plan.kernels,None,plan.workspace)
            } else if kind>=21 {
                (router::prepare(&client,&vs,kind==22,&ints,&floats),None,Vec::new())
            } else if kind==20 {
                let (nodes,extra)=operation_reduce::prepare(&client,&vs,&ints);scratch.extend(extra);
                (nodes,None,Vec::new())
            } else if kind==19 {
                (operation_transform::prepare(&client,&vs,&ints,&floats),None,Vec::new())
            } else if kind>=17 {
                let (nodes,extra)=operation_pointwise::prepare(&client,&vs,kind,&ints,&floats);scratch.extend(extra);
                (nodes,None,Vec::new())
            } else if kind==16 {
                (sequence_scan::prepare_cat(&client,&vs,ints[0] as usize),None,Vec::new())
            } else if kind>=14 {
                let (nodes,extra)=sequence_scan::prepare(&client,&vs[0],&vs[1],kind,ints[0]);scratch.extend(extra);
                (nodes,None,Vec::new())
            } else if kind>=10 {
                let plan=sequence(&client,&vs,kind,&ints,&floats); (plan.kernels,None,plan.workspace)
            } else {let (nodes,update)=prepare(&client,&vs,&scratch,kind,&ints,&floats);(nodes,update,Vec::new())};
            let kernels=prepared.len();let mut pending=prepared.into_iter();let mut graphs=Vec::new();
            while pending.len()>0 {
                let nodes=pending.by_ref().take(4096).collect();
                graphs.push(unsafe{CudaGraph::build(&client,nodes)}.expect("operation graph build failed"));
            }
            let state=State{graphs,client,tensors:vs,scratch,_sequence_workspace:workspace,kind,integers:ints,scalars:floats,kernels,attention_node};
            unsafe {*plan=Box::into_raw(Box::new(NativeOperationGraph{stream:streams::current(),state:Mutex::new(state)}));*result=kernels as u64;}
            return;
        }
        let handle=unsafe{abi::reference(*plan)};assert_eq!(handle.stream,streams::current(),"operation graph queue changed");
        let (outcome,arithmetic)={
            let mut s=handle.state.lock().expect("operation graph state poisoned");
            let outcome=(||->Result<(),ruda::runtime::server::ServerError>{
                match command {
                    1=>{
                        if let Some(index)=s.attention_node {
                            let replacement=attention(&s.client,&s.tensors,&s.scratch,s.kind,&s.integers,&s.scalars,seed,counter);
                            unsafe{s.graphs[0].update_and_replay(index,replacement)}?;
                        } else {for graph in &mut s.graphs {graph.replay()?;}}
                        LAUNCHES.fetch_add(s.kernels as u64,Ordering::Relaxed);finish_dispatch(&s.client);
                        if operation_pointwise::checked_integer(s.kind,&s.integers,&s.tensors)
                            || operation_scatter::checked(s.kind,&s.tensors,&s.integers) {
                            if let Some(graph)=s.graphs.last_mut() {graph.synchronize()?;}
                        }
                    },
                    2=>{if let Some(graph)=s.graphs.last_mut() {graph.synchronize()?;}},
                    3=>unsafe{*result=if let Some(graph)=s.graphs.last_mut() {graph.query()? as u64} else {1};},
                    4=>{for graph in &mut s.graphs {graph.close()?;}},
                    _=>panic!("invalid operation graph command"),
                }
                Ok(())
            })();
            let arithmetic=if outcome.is_ok() && command==1 && operation_pointwise::checked_integer(s.kind,&s.integers,&s.tensors) {
                operation_pointwise::check(&s.client,&s.scratch)
            }else if outcome.is_ok() && command==1 && operation_scatter::checked(s.kind,&s.tensors,&s.integers) {
                operation_scatter::check(&s.client,&s.scratch)
            }else {Ok(())};
            (outcome,arithmetic)
        };
        outcome.expect("native operation graph command failed");
        arithmetic.expect("native checked operation failed");
        if command==4 {unsafe{drop(Box::from_raw(*plan));*plan=std::ptr::null_mut();}}
    })
}
