use super::*;
use super::qwen_kernels as k;
use ruda_driver_cuda::graph::CudaGraph;
use std::sync::Mutex;

pub struct NativeQwenGraph { stream: u64, state: Mutex<State> }
struct State {
    client: ComputeClient<CudaRuntime>, graph: Option<CudaGraph>, tensors: Vec<View>, workspace: Vec<View>,
    dims: Vec<usize>, epsilon: f32, scale: f32, position: usize, kernels: usize,
}

pub(super) fn prepare_linear(c: &ComputeClient<CudaRuntime>, input: &View, weight: &View,
    bias: Option<&View>, output: &View, alpha: f32, beta: f32) -> PreparedKernel<CudaRuntime> {
    let (n, width) = (weight.shape[0], weight.shape[1]);
    let vector = if width % 8 == 0 && [input,weight].iter().all(|v|v.handle.offset_start.unwrap_or(0)%16==0) {8} else {1};
    let args = |v:&View|primitives::tensor(v).into_array_arg();
    macro_rules! run {($ty:ty)=>{unsafe { k::linear::prepare::<$ty,CudaRuntime>(c,
        RudaCount::Static(n.div_ceil(4) as u32,1,1),RudaDim::new_2d(32,4),vector,
        args(input),args(weight),args(bias.unwrap_or(output)),args(output),
        (width/vector) as u32,n as u32,alpha,beta,bias.is_some()) }};}
    match input.dtype {1=>run!(f16),2=>run!(bf16),_=>panic!("Qwen linear needs F16/BF16")}
}

pub(super) fn fused_addmm(bias:&View,a:&View,b:&View,out:&View,alpha:f32,beta:f32)->bool {
    if !matches!(a.dtype,1|2) || a.shape[0]!=1 || a.shape[1]==0 || out.len==0
        || a.strides[1]!=1 || b.strides[0]!=1 || b.strides[1]!=a.shape[1]
        || out.strides[1]!=1 || (bias.len>1 && *bias.strides.last().unwrap()!=1) {return false;}
    let weight=View {handle:b.handle.clone(),shape:vec![b.shape[1],b.shape[0]],
        strides:vec![b.strides[1],b.strides[0]],len:b.len,dtype:b.dtype};
    let c=client();
    let prepared=prepare_linear(&c,a,&weight,if beta==0.0 {None}else {Some(bias)},out,alpha,beta);
    let (task,count,args,origin)=prepared.into_parts();origin.launch(task,count,args);
    finish_dispatch(&c);LAUNCHES.fetch_add(1,Ordering::Relaxed);RUBLAS_CALLS.fetch_add(1,Ordering::Relaxed);true
}

fn prepare_rms(c:&ComputeClient<CudaRuntime>,input:&View,weight:&View,output:&View,epsilon:f32)
    ->PreparedKernel<CudaRuntime> {
    let width=input.len;
    let vector=if width%8==0 && [input,weight,output].iter().all(|v|v.handle.offset_start.unwrap_or(0)%16==0) {8} else {1};
    let args=|v:&View|primitives::tensor(v).into_array_arg();
    macro_rules! run {($ty:ty)=>{unsafe {k::rms_decode::prepare::<$ty,CudaRuntime>(c,
        RudaCount::Static(1,1,1),RudaDim::new_2d(32,4),vector,args(input),args(weight),args(output),
        (width/vector) as u32,width as u32,epsilon)}};}
    match input.dtype {1=>run!(f16),2=>run!(bf16),_=>unreachable!()}
}

fn prepared(s:&State)->Vec<PreparedKernel<CudaRuntime>> {
    let c=&s.client;let t=&s.tensors;let w=&s.workspace;
    let [layers,hidden,intermediate,heads,kv_heads,dim,capacity]=s.dims.as_slice() else {unreachable!()};
    let (layers,hidden,intermediate,heads,kv_heads,dim,capacity)=(*layers,*hidden,*intermediate,*heads,*kv_heads,*dim,*capacity);
    let kv_width=kv_heads*dim;let dtype=t[0].dtype;
    let arg=|v:&View|primitives::tensor(v).into_array_arg();
    let grid=|n:usize|RudaCount::Static(n.div_ceil(128) as u32,1,1);
    let block=RudaDim::new_1d(128);
    let mut nodes=Vec::with_capacity(layers*12+5);
    macro_rules! typed {($ty:ty)=>{{
        nodes.push(unsafe {k::trig::prepare::<$ty,CudaRuntime>(c,grid(dim),block,arg(&t[2]),arg(&w[0]),arg(&w[1]),dim as u32,s.scale)});
        for layer in 0..layers {
            let p=4+layer*9;
            nodes.push(prepare_rms(c,&w[2],&t[p],&w[3],s.epsilon));
            nodes.push(prepare_linear(c,&w[3],&t[p+2],Some(&t[p+3]),&w[4],1.0,1.0));
            nodes.push(unsafe {k::rotary_append::prepare::<$ty,CudaRuntime>(c,grid(hidden+2*kv_width),block,
                arg(&w[4]),arg(&w[1]),arg(&w[5]),arg(&t[p+7]),arg(&t[p+8]),arg(&w[0]),
                hidden as u32,kv_width as u32,dim as u32,capacity as u32)});
            nodes.push(unsafe {k::attention_partial::prepare::<$ty,CudaRuntime>(c,
                RudaCount::Static((heads*64/4) as u32,1,1),RudaDim::new_2d(32,4),
                arg(&w[5]),arg(&t[p+7]),arg(&t[p+8]),arg(&w[0]),arg(&w[6]),arg(&w[7]),
                heads as u32,kv_heads as u32,capacity as u32,((dim as f64).sqrt().recip()) as f32,dim,64)});
            nodes.push(unsafe {k::attention_merge::prepare::<$ty,CudaRuntime>(c,
                RudaCount::Static(heads as u32,1,1),RudaDim::new_2d(32,4),arg(&w[6]),arg(&w[7]),arg(&w[8]),dim,64)});
            nodes.push(prepare_linear(c,&w[8],&t[p+4],None,&w[10],1.0,0.0));
            nodes.push(unsafe {k::residual::prepare::<$ty,CudaRuntime>(c,grid(hidden),block,arg(&w[2]),arg(&w[10]),arg(&w[2]))});
            nodes.push(prepare_rms(c,&w[2],&t[p+1],&w[3],s.epsilon));
            nodes.push(prepare_linear(c,&w[3],&t[p+5],None,&w[11],1.0,0.0));
            nodes.push(unsafe {k::swiglu::prepare::<$ty,CudaRuntime>(c,grid(intermediate),block,arg(&w[11]),arg(&w[12]),intermediate as u32)});
            nodes.push(prepare_linear(c,&w[12],&t[p+6],None,&w[10],1.0,0.0));
            nodes.push(unsafe {k::residual::prepare::<$ty,CudaRuntime>(c,grid(hidden),block,arg(&w[2]),arg(&w[10]),arg(&w[2]))});
        }
        nodes.push(prepare_rms(c,&w[2],&t[0],&w[3],s.epsilon));
        nodes.push(prepare_linear(c,&w[3],&t[1],None,&t[3],1.0,0.0));
    }};}
    match dtype {1=>typed!(f16),2=>typed!(bf16),_=>unreachable!()}
    nodes.push(unsafe {k::advance::prepare::<CudaRuntime>(c,RudaCount::Static(1,1,1),RudaDim::new_1d(1),arg(&w[0]))});
    nodes
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_qwen2_graph(op:u32,plan:*mut *mut NativeQwenGraph,
    descriptors:*const Descriptor,count:usize,integers:*const u64,integer_count:usize,
    floats:*const f32,float_count:usize,position:u64,result:*mut u64)->i32 {
    checked(|| {
        assert!(!plan.is_null() && !result.is_null());unsafe {*result=0;}
        if op==0 {
            assert!(unsafe {(*plan).is_null()});assert_eq!(integer_count,7);assert_eq!(float_count,2);
            let dims:Vec<_>=unsafe {abi::slice(integers,7)}.iter().map(|&x|usize::try_from(x).unwrap()).collect();
            let f=unsafe {abi::slice(floats,2)};
            assert!(dims.iter().all(|&x|x>0 && x<=u32::MAX as usize));
            let (l,h,i,q,kv,d,cap)=(dims[0],dims[1],dims[2],dims[3],dims[4],dims[5],dims[6]);
            assert_eq!(h,q.checked_mul(d).unwrap());assert_eq!(q%kv,0);assert_eq!(d%2,0);assert!((position as usize)<cap);
            assert!(f.iter().all(|x|x.is_finite() && *x>0.0));assert_eq!(count,4+l*9);
            let tensors:Vec<_>=unsafe {abi::slice(descriptors,count)}.iter().map(|x|unsafe {abi::view(x)}).collect();
            let dtype=tensors[0].dtype;assert!(matches!(dtype,1|2));
            for (n,t) in tensors.iter().enumerate() {operation_graph::packed(t);assert_eq!(t.dtype,if n==2 {0}else {dtype});}
            assert_eq!(tensors[0].shape,vec![h]);assert_eq!(tensors[1].shape.len(),2);assert_eq!(tensors[1].shape[1],h);
            assert_eq!(tensors[2].shape,vec![d/2]);assert_eq!(tensors[3].shape,vec![1,1,tensors[1].shape[0]]);
            for layer in 0..l {
                let p=4+layer*9;
                for n in [p,p+1] {assert_eq!(tensors[n].shape,vec![h]);}
                assert_eq!(tensors[p+2].shape,vec![h+2*kv*d,h]);assert_eq!(tensors[p+3].shape,vec![h+2*kv*d]);
                assert_eq!(tensors[p+4].shape,vec![h,h]);assert_eq!(tensors[p+5].shape,vec![2*i,h]);
                assert_eq!(tensors[p+6].shape,vec![h,i]);
                for n in [p+7,p+8] {assert_eq!(tensors[n].shape,vec![1,kv,cap,d]);}
            }
            let c=client().fixed_execution_queue();
            let allocate=|shape:Vec<usize>,kind:u32| {
                let len=shape.iter().try_fold(1usize,|a,&b|a.checked_mul(b)).unwrap();
                assert!(len<=u32::MAX as usize);View::packed(c.empty(len*element_bytes(kind)),shape,kind)
            };
            let position_handle=c.create_from_slice(&(position as u32).to_ne_bytes());
            let workspace=vec![View::packed(position_handle,vec![1],5),allocate(vec![2*d],dtype),
                allocate(vec![1,h],dtype),allocate(vec![1,h],dtype),allocate(vec![1,h+2*kv*d],dtype),
                allocate(vec![h],dtype),allocate(vec![q*64,d],0),allocate(vec![q*64,2],0),
                allocate(vec![1,h],dtype),allocate(vec![q,2],0),allocate(vec![1,h],dtype),
                allocate(vec![1,2*i],dtype),allocate(vec![1,i],dtype)];
            let mut state=State{client:c,graph:None,tensors,workspace,dims,epsilon:f[0],scale:f[1],
                position:position as usize,kernels:0};
            let nodes=prepared(&state);state.kernels=nodes.len();
            state.graph=Some(unsafe {CudaGraph::build(&state.client,nodes)}.expect("Qwen decoder graph build failed"));
            unsafe {*result=state.kernels as u64;*plan=Box::into_raw(Box::new(NativeQwenGraph{stream:streams::current(),state:Mutex::new(state)}));}
            static_graph::BUILDS.fetch_add(1,Ordering::Relaxed);return;
        }
        if op==3 && unsafe {(*plan).is_null()} {return;}
        let native=unsafe {abi::reference(*plan)};
        if matches!(op,1|4) {assert_eq!(native.stream,streams::current());}
        {
            let mut s=native.state.lock().expect("Qwen graph lock poisoned");
            match op {
                1|4=>{
                    assert_eq!(s.position,position as usize);assert!(s.position<s.dims[6]);assert_eq!(count,1);
                    let input=unsafe {abi::view(descriptors)};assert_eq!(input.len,s.dims[1]);assert_eq!(input.dtype,s.tensors[0].dtype);
                    operation_graph::packed(&input);
                    let reshaped=View::packed(input.handle,s.workspace[2].shape.clone(),input.dtype);
                    convert(&reshaped,&s.workspace[2]);
                    if op==1 {s.graph.as_mut().unwrap().replay().expect("Qwen graph replay failed");static_graph::REPLAYS.fetch_add(1,Ordering::Relaxed);}
                    else {for node in prepared(&s) {let (task,count,args,origin)=node.into_parts();origin.launch(task,count,args);}}
                    s.position+=1;LAUNCHES.fetch_add(s.kernels as u64,Ordering::Relaxed);
                    RUBLAS_CALLS.fetch_add((s.dims[0]*4+1) as u64,Ordering::Relaxed);
                    FUSED_RMS_NORM_CALLS.fetch_add((s.dims[0]*2+1) as u64,Ordering::Relaxed);finish_dispatch(&s.client);
                }
                2=>s.graph.as_mut().unwrap().synchronize().expect("Qwen graph sync failed"),
                3=>s.graph.as_mut().unwrap().close().expect("Qwen graph close failed"),
                _=>panic!("unknown Qwen graph command"),
            }
        }
        if op==3 {unsafe {drop(Box::from_raw(*plan));*plan=std::ptr::null_mut();}}
    })
}
