//! Fixed-address inference subgraphs for native PyTorch tensors.
//! Reuses production pointwise/matrix/reduction/normalization kernels and the existing RUDA CudaGraph.
//! No stream capture, CPU fallback, or CUDA C++ compiler.
//! SILU_MUL fuses two operators while preserving the intermediate storage cast.
use super::*;
use super::graph_contract::{self, NodeSpec, TensorSpec, RMS_NORM, SILU_MUL, NO_WEIGHT};
use ruda_driver_cuda::graph::CudaGraph;
use ruda_driver_cuda::interop::{self,StreamCommand};
use std::sync::Mutex;

pub(super) static BUILDS: AtomicU64 = AtomicU64::new(0);
pub(super) static REPLAYS: AtomicU64 = AtomicU64::new(0);
pub(super) static EAGER_RUNS: AtomicU64 = AtomicU64::new(0);

struct State {
    graph: Option<CudaGraph>,
    empty_completion: Option<u64>,
    client: ComputeClient<CudaRuntime>,
    stream: u64,
    tensors: Vec<View>,
    nodes: Vec<NodeSpec>,
    inputs: usize,
    indexing: super::graph_index::IndexBuffers,
    kernels: usize,
}
pub struct NativeStaticGraph { stream: u64, tracked: bool, state: Mutex<State> }

impl State {
    fn stream_command(&self,command:StreamCommand)->Result<u64,ruda::runtime::server::ServerError> {
        interop::command(self.client.device_id(),command)
    }
    fn synchronize(&mut self)->Result<(),ruda::runtime::server::ServerError> {
        if let Some(graph)=&mut self.graph {graph.synchronize()}
        else {self.stream_command(StreamCommand::Synchronize(self.stream)).map(|_|())}
    }
    fn replay(&mut self)->Result<(),ruda::runtime::server::ServerError> {
        if let Some(graph)=&mut self.graph {graph.replay()?;}
        else if let Some(event)=self.empty_completion {
            self.stream_command(StreamCommand::Record{stream:self.stream,event,timing:false})?;
        }
        Ok(())
    }
    fn query(&mut self)->Result<bool,ruda::runtime::server::ServerError> {
        if let Some(graph)=&mut self.graph {graph.query()}
        else {self.stream_command(StreamCommand::Query(self.stream)).map(|value|value!=0)}
    }
    fn query_completion(&mut self)->Result<bool,ruda::runtime::server::ServerError> {
        if let Some(graph)=&mut self.graph {graph.query_completion()}
        else {self.stream_command(StreamCommand::EventQuery(self.empty_completion.unwrap())).map(|value|value!=0)}
    }
    fn wait_completion(&mut self)->Result<(),ruda::runtime::server::ServerError> {
        if let Some(graph)=&mut self.graph {graph.wait_completion()}
        else {self.stream_command(StreamCommand::EventSynchronize(self.empty_completion.unwrap())).map(|_|())}
    }
    fn close(&mut self)->Result<(),ruda::runtime::server::ServerError> {
        if let Some(graph)=&mut self.graph {graph.close()?;}
        else {
            self.synchronize()?;
            if let Some(event)=self.empty_completion {
                self.stream_command(StreamCommand::EventDestroy(event))?;
                self.empty_completion=None;
            }
        }
        Ok(())
    }
}

fn prepare(state_client: &ComputeClient<CudaRuntime>, ts: &[View], n: &NodeSpec, output: usize)
    -> PreparedKernel<CudaRuntime>
{
    if let Some(kernel)=super::graph_factory::prepare(state_client,ts,n,output) {return kernel;}
    let a = &ts[n.a as usize];
    let out = &ts[output];
    let b = if n.b == NO_WEIGHT { a } else { &ts[n.b as usize] };
    if let Some(kernel)=super::graph_index::prepare_layout(state_client,a,out,n) {return kernel;}
    if matches!(n.op,7|30) {
        if let Some(kernel)=super::graph_matmul::prepare(state_client,a,b,out) {return kernel;}
    }
    macro_rules! run {
        ($dtype:ty) => {{
            if n.op == SILU_MUL {
                let blocks = u32::try_from(out.len.div_ceil(128)).expect("graph SiLU-mul grid overflow");
                unsafe { kernels::static_silu_mul::prepare::<$dtype, CudaRuntime>(
                    state_client, RudaCount::Static(blocks,1,1), RudaDim::new_1d(128),
                    a.arg(), b.arg(), out.arg()) }
            } else if n.op == 112 || n.op == 114 {
                let blocks = u32::try_from(out.len.div_ceil(128)).expect("graph layout grid overflow");
                macro_rules! cast { ($out:ty) => { unsafe { kernels::convert::prepare::<$dtype,$out,CudaRuntime>(
                    state_client,RudaCount::Static(blocks,1,1),RudaDim::new_1d(128),a.arg(),out.arg()) } }; }
                match out.dtype { 0=>cast!(f32),1=>cast!(f16),2=>cast!(bf16),_=>unreachable!() }
            } else if n.op == 113 {
                let blocks = u32::try_from(out.len.div_ceil(128)).expect("graph permutation grid overflow");
                unsafe { kernels::graph_permute::prepare::<$dtype,CudaRuntime>(state_client,
                    RudaCount::Static(blocks,1,1),RudaDim::new_1d(128),a.arg(),out.arg(),n.scalar as u32) }
            } else if n.op == 115 {
                let blocks = u32::try_from(out.len.div_ceil(128)).expect("graph expansion grid overflow");
                unsafe { kernels::graph_expand::prepare::<$dtype,CudaRuntime>(state_client,
                    RudaCount::Static(blocks,1,1),RudaDim::new_1d(128),a.arg(),out.arg()) }
            } else if matches!(n.op,57 | 116..=118 | 121..=124) {
                let blocks = u32::try_from(out.len.div_ceil(128)).expect("graph broadcast grid overflow");
                unsafe { kernels::graph_broadcast::prepare::<$dtype,CudaRuntime>(state_client,
                    RudaCount::Static(blocks,1,1),RudaDim::new_1d(128),a.arg(),b.arg(),out.arg(),n.scalar,n.op) }
            } else if matches!(n.op,129..=133) {
                let blocks = u32::try_from(out.len.div_ceil(128)).expect("graph scalar math grid overflow");
                unsafe { kernels::graph_scalar_math::prepare::<$dtype,CudaRuntime>(state_client,
                    RudaCount::Static(blocks,1,1),RudaDim::new_1d(128),a.arg(),out.arg(),n.scalar,n.op) }
            } else if n.op == 119 || n.op == 120 {
                let mask = n.scalar as u32;
                let shape = a.shape.iter().enumerate().map(|(d,&size)|if mask&(1<<d)!=0 {1} else {size}).collect();
                let expanded = View::packed(out.handle.clone(),shape,out.dtype);
                let blocks = u32::try_from(out.len.div_ceil(128)).expect("graph reduction grid overflow");
                unsafe { kernels::reduce_sum_storage::prepare::<$dtype,$dtype,CudaRuntime>(state_client,
                    RudaCount::Static(blocks,1,1),RudaDim::new_1d(128),a.arg(),expanded.arg(),n.op==120) }
            } else if n.op == RMS_NORM {
                let rows = a.len / a.shape[a.shape.len()-1];
                let blocks = u32::try_from((rows * 32).div_ceil(128)).expect("graph RMSNorm grid overflow");
                unsafe { kernels::rms_norm_warp::prepare::<$dtype, CudaRuntime>(
                    state_client, RudaCount::Static(blocks,1,1), RudaDim::new_1d(128),
                    a.arg(), b.arg(), out.arg(), n.scalar, n.b != NO_WEIGHT) }
            } else if n.op == 7 || n.op == 30 {
                let blocks = u32::try_from(out.len.div_ceil(128)).expect("graph matmul grid overflow");
                unsafe { kernels::matmul_storage::prepare::<$dtype, $dtype, CudaRuntime>(
                    state_client, RudaCount::Static(blocks,1,1), RudaDim::new_1d(128),
                    a.arg(), b.arg(), out.arg(), n.op == 30) }
            } else if n.op == 104 || n.op == 105 {
                let blocks = u32::try_from(out.len.div_ceil(128)).expect("graph reduction grid overflow");
                unsafe { kernels::reduce_sum_storage::prepare::<$dtype, $dtype, CudaRuntime>(
                    state_client, RudaCount::Static(blocks,1,1), RudaDim::new_1d(128),
                    a.arg(), out.arg(), n.op == 105) }
            } else if matches!(n.op, 102 | 103 | 106 | 107) {
                let axis = n.scalar as usize;
                let rows = a.len / a.shape[axis];
                let blocks = u32::try_from(rows.div_ceil(128)).expect("graph softmax grid overflow");
                unsafe { kernels::softmax::prepare::<$dtype, $dtype, CudaRuntime>(
                    state_client, RudaCount::Static(blocks,1,1), RudaDim::new_1d(128),
                    a.arg(), b.arg(), out.arg(), axis, n.op >= 106, n.op == 103 || n.op == 107) }
            } else {
                let blocks = u32::try_from(out.len.div_ceil(128)).expect("graph pointwise grid overflow");
                unsafe { kernels::pointwise::prepare::<$dtype,$dtype,$dtype,CudaRuntime>(
                    state_client, RudaCount::Static(blocks,1,1), RudaDim::new_1d(128),
                    a.arg(), b.arg(), out.arg(), n.scalar, n.op) }
            }
        }};
    }
    match a.dtype { 0 => run!(f32), 1 => run!(f16), 2 => run!(bf16), _ => unreachable!() }
}

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_graph_api_version() -> u32 { 3 }

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_graph_layout_api_version() -> u32 { 1 }

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_graph_math_api_version() -> u32 { 1 }

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_graph_index_api_version() -> u32 { 1 }

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_graph_factory_api_version() -> u32 { 1 }

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_graph_empty_api_version() -> u32 { 1 }

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_graph_root_api_version() -> u32 { 1 }

fn prepare_all(client:&ComputeClient<CudaRuntime>,tensors:&[View],nodes:&[NodeSpec],inputs:usize,
    indexing:&super::graph_index::IndexBuffers)->Vec<PreparedKernel<CudaRuntime>> {
    let mut kernels=indexing.preamble(client);
    for (i,node) in nodes.iter().enumerate() {
        if let Some(indexed)=indexing.prepare(client,tensors,node,i,inputs) {kernels.extend(indexed);}
        else if tensors[inputs+i].len!=0 {kernels.push(prepare(client,tensors,node,inputs+i));}
    }
    kernels
}

/// Optional in-process extension API 3; base tensor ABI remains 10.
/// 0=build, 1=replay, 2=wait, 3=query fixed queue, 4=close,
/// 5=preallocated eager control (same kernels, one final sync policy),
/// 6=query tracked completion, 7=wait tracked completion.
/// Flags: bit0 completion event, bit1 infer memory dependencies. No defaults changed.
///
/// # Safety
/// The C++ owner retains all descriptors/allocations and serializes access.
/// Input/output aliasing is forbidden. Writable slots may reuse an EXACT buffer
/// only after the earlier value's final consumer; reuse inside one kernel is
/// forbidden. The C++ owner checks all live ranges before build. Readonly input
/// aliasing is permitted. Shapes, dtypes and addresses are fixed. External
/// mutations obey the same queue and no tensor requires gradients. Destroy once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_graph(
    op:u32, plan:*mut *mut NativeStaticGraph, descriptors:*const Descriptor, tensor_count:usize,
    nodes:*const NodeSpec, node_count:usize, input_count:usize, flags:u32, result:*mut u64,
) -> i32 {
    checked(|| {
        assert!(!plan.is_null() && !result.is_null(), "null static graph ABI output");
        unsafe { *result=0; }
        if op==0 {
            assert!(unsafe{(*plan).is_null()},"static graph already built");
            assert!(flags & !3 == 0, "unknown static graph flags");
            assert!(!descriptors.is_null() && !nodes.is_null(), "null static graph build input");
            assert!((1..=graph_contract::MAX_TENSORS).contains(&tensor_count));
            assert!((1..=graph_contract::MAX_NODES).contains(&node_count));
            let descriptors=unsafe{crate::abi::slice(descriptors,tensor_count)};
            let nodes=unsafe{crate::abi::slice(nodes,node_count)}.to_vec();
            let tensors:Vec<View>=descriptors.iter().map(|d| unsafe{crate::abi::view(d)}).collect();
            let specs:Vec<_>=tensors.iter().zip(descriptors).map(|(t,d)|TensorSpec{
                shape:if d.rank==0 {&[]} else {&t.shape},
                strides:if d.rank==0 {&[]} else {&t.strides},dtype:t.dtype}).collect();
            graph_contract::validate(&specs,&nodes,input_count).expect("invalid native static graph");
            let client=client().fixed_execution_queue();
            let indexing=super::graph_index::IndexBuffers::new(&client,&tensors,&nodes,input_count);
            let prepared=prepare_all(&client,&tensors,&nodes,input_count,&indexing);
            let kernels=prepared.len();
            // SAFETY: descriptor contracts checked above and storage overlap by C++;
            // all registered kernels have explicit bindings and no hidden side effects.
            let graph=if prepared.is_empty() {None} else {Some(unsafe {
                match flags {
                    0=>CudaGraph::build(&client,prepared),
                    1=>CudaGraph::build_tracked(&client,prepared),
                    2=>CudaGraph::build_inferred(&client,prepared),
                    3=>CudaGraph::build_inferred_tracked(&client,prepared),
                    _=>unreachable!(),
                }
            }.expect("native static graph build failed"))};
            let stream=client.execution_stream().value;
            let empty_completion=if graph.is_none() && flags&1!=0 {
                Some(interop::command(client.device_id(),StreamCommand::Record{stream,event:0,timing:false})
                    .expect("empty graph completion event creation failed"))
            } else {None};
            let edges=graph.as_ref().map_or(0,|graph|graph.edge_count() as u64);
            let state=State{graph,empty_completion,client,stream,tensors,nodes,inputs:input_count,indexing,kernels};
            unsafe { *plan=Box::into_raw(Box::new(NativeStaticGraph{stream:state.stream,tracked:flags&1!=0,state:Mutex::new(state)})); *result=edges; }
            BUILDS.fetch_add(1,Ordering::Relaxed);
            return;
        }
        if op==4 && unsafe{(*plan).is_null()} { return; }
        assert!(!unsafe{(*plan).is_null()},"static graph is closed");
        let native=unsafe{&**plan};
        assert!((1..=7).contains(&op), "unknown static graph command");
        assert!(!matches!(op,6|7) || native.tracked,"tracked static graph required");
        if op==1 || op==5 {
            assert_eq!(native.stream,streams::current(),"static graph replay must use its creation stream");
        }
        let command_result = {
            let mut state=native.state.lock().expect("static graph lock poisoned");
            // Return ordinary driver errors before dropping the lock, THEN
            // translate at the ABI boundary. A retryable close failure must not
            // poison the mutex merely because expect() ran while holding it.
            let result=(|| -> Result<(), ruda::runtime::server::ServerError> {
            match op {
                1 | 5 => {
                    if op==1 {
                        state.replay()?;
                        REPLAYS.fetch_add(1,Ordering::Relaxed);
                    } else {
                        for prepared in prepare_all(&state.client,&state.tensors,&state.nodes,state.inputs,&state.indexing) {
                            let (task,count,args,origin)=prepared.into_parts();
                            origin.launch(task,count,args);
                        }
                        EAGER_RUNS.fetch_add(1,Ordering::Relaxed);
                    }
                    LAUNCHES.fetch_add(state.kernels as u64,Ordering::Relaxed);
                    finish_dispatch(&state.client);
                    if state.indexing.enabled() {state.synchronize()?;}
                }
                2=>state.synchronize()?,
                3=>unsafe{*result=state.query()? as u64;},
                4=>state.close()?,
                6=>unsafe{*result=state.query_completion()? as u64;},
                7=>state.wait_completion()?,
                _=>panic!("unknown static graph command"),
            }
            Ok(())
            })();
            let indexing_result=if result.is_ok() && matches!(op,1|5) {state.indexing.check(&state.client)} else {Ok(())};
            (result,indexing_result)
        };
        command_result.0.expect("native static graph command failed");
        command_result.1.expect("native static graph indexing failed");
        if op==4 { unsafe{drop(Box::from_raw(*plan)); *plan=std::ptr::null_mut();} }
    })
}
