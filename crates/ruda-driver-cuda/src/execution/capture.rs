use super::graph::error;
use cudarc::driver::sys::*;
use ruda::runtime::{memory_management::{ManagedMemoryBinding, ManagedMemoryId, PageGuard, WeakMemoryBinding}, server::{Binding, Handle, ServerError}};
use ruda_core::{bytes::Bytes, stream_id::StreamId};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Default, Debug)]
pub(crate) struct CaptureState {
    pub owner: Option<StreamId>,
    pub root: Option<StreamId>,
    pub warming: bool,
    pub recording: bool,
    pub entry_cursor: u64,
    pub metadata: HashMap<Vec<u8>, Handle>,
    pub primed: Vec<ManagedMemoryBinding>,
    pub host: Vec<Bytes>,
    pub retained: Vec<Handle>,
    pub failed: Option<ServerError>,
    pub written: HashMap<(ManagedMemoryId, u64, u64), CapturedWrite>,
    pub children: Vec<Arc<Captured>>,
}

impl CaptureState {
    pub fn absorb(&mut self, other: Self) {
        if self.failed.is_none() { self.failed = other.failed; }
        self.host.extend(other.host);
        self.retained.extend(other.retained);
        self.retained.extend(other.metadata.into_values());
        self.primed.extend(other.primed);
        self.children.extend(other.children);
        self.written.extend(other.written);
    }

    pub fn record_write(&mut self, binding: &Binding) {
        if self.recording {
            let start = binding.offset_start.unwrap_or(0);
            let end = binding.size - binding.offset_end.unwrap_or(0);
            self.written.entry((binding.memory.id(), start, end)).or_insert_with(|| CapturedWrite {
                memory: binding.memory.downgrade(), start, end,
            });
        }
    }
}

#[derive(Default, Debug)]
pub(crate) struct Captures {
    pub preparing: HashSet<StreamId>,
    pub active: HashSet<StreamId>,
    pub members: HashMap<StreamId, StreamId>,
    pub warm_members: HashMap<StreamId, StreamId>,
    pub communication: HashMap<StreamId, CaptureCommunication>,
    pub next: u64,
    pub graphs: HashMap<u64, Arc<Captured>>,
    pub deferred_releases: Vec<u64>,
}

#[derive(Debug)]
pub(crate) struct Captured {
    pub executable: usize,
    pub context: usize,
    pub stream: usize,
    pub stream_id: StreamId,
    pub pages: Vec<PageGuard>,
    pub host: Vec<Bytes>,
    pub metadata: Vec<Handle>,
    pub retained: Vec<Handle>,
    pub written: Vec<CapturedWrite>,
    pub children: Vec<Arc<Captured>>,
    pub communication: Option<Arc<CaptureCommunicationStream>>,
}

#[derive(Debug)]
pub(crate) struct CaptureCommunication {
    pub stream: Arc<CaptureCommunicationStream>,
    pub joined: bool,
}

#[derive(Debug)]
pub(crate) struct CaptureCommunicationStream {
    pub raw: usize,
    context: usize,
}

impl CaptureCommunicationStream {
    pub fn new(context: usize) -> Result<Arc<Self>, ServerError> {
        let mut stream = std::ptr::null_mut();
        checked("cuStreamCreate capture communication", unsafe {
            cuStreamCreate(&mut stream, CUstream_flags::CU_STREAM_NON_BLOCKING as u32)
        })?;
        Ok(Arc::new(Self { raw: stream as usize, context }))
    }
}

impl Drop for CaptureCommunicationStream {
    fn drop(&mut self) {
        unsafe {
            let mut previous = std::ptr::null_mut();
            if cuCtxGetCurrent(&mut previous) != CUresult::CUDA_SUCCESS { return; }
            if cuCtxSetCurrent(self.context as CUcontext) != CUresult::CUDA_SUCCESS { return; }
            let _ = cuStreamSynchronize(self.raw as CUstream);
            let _ = cuStreamDestroy_v2(self.raw as CUstream);
            let _ = cuCtxSetCurrent(previous);
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CapturedWrite {
    pub memory: WeakMemoryBinding,
    pub start: u64,
    pub end: u64,
}

pub(crate) fn fail_written(written: impl IntoIterator<Item = CapturedWrite>, error: &ServerError) {
    let mut failure = None;
    for binding in written {
        if binding.start < binding.end {
            if let Some(memory) = binding.memory.upgrade() {
                let root = failure.get_or_insert_with(|| ruda::runtime::memory_management::ErrorGraph::new(error.clone()));
                memory.claim(binding.start, binding.end, root.clone());
            }
        }
    }
}

impl Drop for Captured {
    fn drop(&mut self) {
        unsafe {
            let mut previous = std::ptr::null_mut();
            let _ = cuCtxGetCurrent(&mut previous);
            let _ = cuCtxSetCurrent(self.context as CUcontext);
            let _ = cuStreamSynchronize(self.stream as CUstream);
            let _ = cuGraphExecDestroy(self.executable as CUgraphExec);
            let _ = cuCtxSetCurrent(previous);
        }
    }
}

pub(crate) fn checked(operation: &str, status: CUresult) -> Result<(), ServerError> {
    super::status::checked(operation, status)
}

pub(crate) fn dependency(source: CUstream, destination: CUstream) -> Result<(), ServerError> {
    unsafe {
        let mut event = std::ptr::null_mut();
        checked("cuEventCreate capture dependency", cuEventCreate(&mut event, CUevent_flags::CU_EVENT_DISABLE_TIMING as u32))?;
        let result = checked("cuEventRecord capture dependency", cuEventRecord(event, source))
            .and_then(|_| checked("cuStreamWaitEvent capture dependency", cuStreamWaitEvent(destination, event, 0)));
        let destroyed = checked("cuEventDestroy capture dependency", cuEventDestroy_v2(event));
        result.and(destroyed)
    }
}

pub(crate) unsafe fn instantiate(stream: CUstream, failed: Option<ServerError>) -> Result<CUgraphExec, ServerError> {
    unsafe {
        let mut graph = std::ptr::null_mut();
        let ended = cuStreamEndCapture(stream, &mut graph);
        if ended != CUresult::CUDA_SUCCESS {
            if !graph.is_null() { let _ = cuGraphDestroy(graph); }
            checked("cuStreamEndCapture", ended)?;
        }
        let result = (|| {
            if let Some(failed) = failed { return Err(failed); }
            let mut count = 0;
            checked("cuGraphGetNodes", cuGraphGetNodes(graph, std::ptr::null_mut(), &mut count))?;
            let mut nodes = vec![std::ptr::null_mut(); count];
            checked("cuGraphGetNodes", cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut count))?;
            for node in nodes.into_iter().take(count) {
                let mut kind = CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL;
                checked("cuGraphNodeGetType", cuGraphNodeGetType(node, &mut kind))?;
                if matches!(kind, CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_ALLOC | CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_FREE) {
                    return Err(error("Captured allocation/free nodes cannot be replayed with stable buffers"));
                }
            }
            let mut executable = std::ptr::null_mut();
            checked("cuGraphInstantiateWithFlags", cuGraphInstantiateWithFlags(&mut executable, graph, 0))?;
            Ok(executable)
        })();
        if !graph.is_null() { let _ = cuGraphDestroy(graph); }
        result
    }
}
