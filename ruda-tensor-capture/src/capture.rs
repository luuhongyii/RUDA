use alloc::{boxed::Box, collections::{BTreeMap, BTreeSet}, format, string::String, sync::Arc, vec::Vec};
use core::sync::atomic::{AtomicU64, Ordering};
use ruda_core::{device::Device, future::DynFut};
use ruda_tensor::{BoolStore, DType, DTypeUsage, DTypeUsageSet, DeviceId, DeviceOps, DeviceSettings,
    ExecutionError, Shape, TensorData};
use ruda_tensor::graph::{GraphBindings, GraphId, GraphIr, IrVisitorMut, OperationIr, TensorId, TensorIr, TensorStatus};
use ruda_tensor_router::{CachedGraph, MultiBackendBridge, RunnerChannel, RunnerClient,
    RunnerClientRegistration, RouterTensor, register_scoped_client};
use spin::Mutex;

static DEVICE_COUNTER: AtomicU64 = AtomicU64::new(0);
static TENSOR_COUNTER: AtomicU64 = AtomicU64::new(0);
static GRAPH_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Records tensor operations without executing kernels.
pub type CaptureBackend = ruda_tensor_router::BackendRouter<CaptureChannel>;

/// Reusable logical device; each scope has an isolated tensor and graph namespace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaptureDevice { id: u16 }

impl CaptureDevice {
    /// Record a scope with explicit inputs and outputs. Completion closes it immediately.
    pub fn capture_scope(&self, capture: impl FnOnce(CaptureScope) -> CompletedCaptureScope)
        -> Result<CapturedGraph, CaptureError> {
        let session = Arc::new(CaptureSession::default());
        let client = CaptureClient { device: *self, session: session.clone() };
        let registration = register_scoped_client::<CaptureChannel>(self, client.clone())
            .ok_or(CaptureError::AlreadyActive)?;
        let guard = ScopeGuard { client, _registration: registration };
        guard.finish(capture(CaptureScope { session }))
    }
}

impl Default for CaptureDevice {
    fn default() -> Self {
        let id = DEVICE_COUNTER.fetch_add(1, Ordering::Relaxed);
        assert!(id <= u16::MAX as u64, "Capture device identifier space exhausted");
        Self { id: id as u16 }
    }
}

impl Device for CaptureDevice {
    fn from_id(id: DeviceId) -> Self {
        assert_eq!(id.type_id, 0, "Invalid capture device type");
        Self { id: id.index_id }
    }
    fn to_id(&self) -> DeviceId { DeviceId::new(0, self.id) }
}

impl DeviceOps for CaptureDevice {
    fn defaults(&self) -> Option<DeviceSettings> {
        Some(DeviceSettings::new(DType::F32, DType::I64, DType::Bool(BoolStore::U8)))
    }
}

/// Concrete operation descriptors and retained initialization data, not an executable GPU graph.
#[derive(Clone, Debug)]
pub struct CapturedGraph {
    pub graph: GraphIr,
    pub values: BTreeMap<TensorId, TensorData>,
}

#[derive(Debug)]
pub struct CaptureScope { session: Arc<CaptureSession> }

impl CaptureScope {
    /// Declare ordered runtime inputs and surviving outputs, then close the scope.
    pub fn complete(self, inputs: impl IntoIterator<Item = TensorId>, outputs: impl IntoIterator<Item = TensorId>)
        -> CompletedCaptureScope {
        let inputs = inputs.into_iter().collect();
        let outputs = outputs.into_iter().collect();
        self.session.state.lock().closed = true;
        CompletedCaptureScope { inputs, outputs }
    }
}

#[derive(Debug)]
pub struct CompletedCaptureScope { inputs: Vec<TensorId>, outputs: Vec<TensorId> }

#[derive(Debug, Default)]
struct CaptureSession { state: Mutex<CaptureState> }

struct ScopeGuard {
    client: CaptureClient,
    _registration: RunnerClientRegistration<CaptureChannel>,
}

impl ScopeGuard {
    fn finish(self, scope: CompletedCaptureScope) -> Result<CapturedGraph, CaptureError> {
        let mut state = self.client.session.state.lock();
        let inputs: Vec<_> = scope.inputs.into_iter().map(|id| state.resolve_alias(id)).collect();
        let outputs: Vec<_> = scope.outputs.into_iter().map(|id| state.resolve_alias(id)).collect();
        let known: BTreeSet<_> = state.operations.iter().flat_map(OperationIr::nodes)
            .map(|tensor| tensor.id).chain(state.values.keys().copied()).collect();
        for (boundary, ids) in [("input", &inputs), ("output", &outputs)] {
            let mut unique = BTreeSet::new();
            if let Some(&tensor) = ids.iter().find(|id| !unique.insert(**id)) {
                return Err(CaptureError::DuplicateBoundary { boundary, tensor });
            }
            if let Some(&tensor) = ids.iter().find(|id| !known.contains(*id)) {
                return Err(CaptureError::UnknownBoundary { boundary, tensor });
            }
        }
        let (inferred_inputs, inferred_outputs) = GraphIr::classify(&state.operations);
        for &tensor in &inputs {
            if !inferred_inputs.contains(&tensor) && !state.values.contains_key(&tensor) {
                return Err(CaptureError::InvalidInput { tensor });
            }
        }
        for &tensor in &inferred_inputs {
            if !inputs.contains(&tensor) && !state.values.contains_key(&tensor) {
                return Err(CaptureError::UndeclaredInput { tensor });
            }
        }
        for &tensor in &outputs {
            let consumed = state.operations.iter().any(|op| op.nodes().into_iter()
                .any(|node| node.id == tensor && node.status == TensorStatus::ReadWrite));
            if consumed || (!inferred_inputs.contains(&tensor) && !inferred_outputs.contains(&tensor)) {
                return Err(CaptureError::InvalidOutput { tensor });
            }
        }
        Ok(CapturedGraph {
            graph: GraphIr { operations: core::mem::take(&mut state.operations), inputs, outputs },
            values: state.values.clone(),
        })
    }
}

impl Drop for ScopeGuard {
    fn drop(&mut self) { self.client.session.state.lock().closed = true; }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CaptureError {
    AlreadyActive,
    InvalidDevice,
    UnknownBoundary { boundary: &'static str, tensor: TensorId },
    DuplicateBoundary { boundary: &'static str, tensor: TensorId },
    UndeclaredInput { tensor: TensorId },
    InvalidInput { tensor: TensorId },
    InvalidOutput { tensor: TensorId },
}

impl core::fmt::Display for CaptureError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AlreadyActive => f.write_str("A capture scope is already active on this device"),
            Self::InvalidDevice => f.write_str("A capture scope requires a capture device"),
            Self::UnknownBoundary { boundary, tensor } => write!(f, "Unknown graph {boundary} tensor {tensor}"),
            Self::DuplicateBoundary { boundary, tensor } => write!(f, "Duplicate graph {boundary} tensor {tensor}"),
            Self::UndeclaredInput { tensor } => write!(f, "Graph reads undeclared input tensor {tensor}"),
            Self::InvalidInput { tensor } => write!(f, "Graph input {tensor} is produced inside the graph"),
            Self::InvalidOutput { tensor } => write!(f, "Graph output {tensor} does not survive the graph"),
        }
    }
}

#[derive(Clone)]
pub struct CaptureChannel;

impl RunnerChannel for CaptureChannel {
    type Device = CaptureDevice;
    type Bridge = CaptureBridge;
    type Client = CaptureClient;
    type FloatElem = f32;
    type IntElem = i64;
    type BoolElem = u8;
    fn name(_: &CaptureDevice) -> String { "capture".into() }
    fn init_client(_: &CaptureDevice) -> CaptureClient {
        panic!("Capture tensor operations require an active CaptureDevice::capture_scope")
    }
    fn get_tensor_handle(tensor: &TensorIr, client: &CaptureClient) -> TensorData {
        client.session.state.lock().value(tensor.id)
            .unwrap_or_else(|| panic!("Captured tensor {} has no initialized value", tensor.id))
    }
    fn register_tensor(client: &CaptureClient, data: TensorData, _: Shape, _: DType) -> RouterTensor<CaptureClient> {
        client.register_tensor_data(data)
    }
}

/// Carries only initialized values between capture devices; computed tensors have no data.
pub struct CaptureBridge;
impl MultiBackendBridge for CaptureBridge {
    type TensorHandle = TensorData;
    type Device = CaptureDevice;
    fn change_backend_float(data: TensorData, _: Shape, _: &CaptureDevice) -> TensorData { data }
    fn change_backend_int(data: TensorData, _: Shape, _: &CaptureDevice) -> TensorData { data }
    fn change_backend_bool(data: TensorData, _: Shape, _: &CaptureDevice) -> TensorData { data }
    fn change_backend_quantized(_: TensorData, _: Shape, _: &CaptureDevice) -> TensorData {
        panic!("Quantized capture tensors are not supported")
    }
}

#[derive(Debug, Default)]
struct CaptureState {
    operations: Vec<OperationIr>,
    values: BTreeMap<TensorId, TensorData>,
    graphs: BTreeMap<GraphId, Arc<CachedGraph>>,
    aliases: BTreeMap<TensorId, TensorId>,
    closed: bool,
}

impl CaptureState {
    fn assert_open(&self) { assert!(!self.closed, "Capture scope is already complete"); }
    fn resolve_alias(&self, mut id: TensorId) -> TensorId {
        while let Some(source) = self.aliases.get(&id) { id = *source; }
        id
    }
    fn value(&self, id: TensorId) -> Option<TensorData> { self.values.get(&self.resolve_alias(id)).cloned() }
    fn register_op(&mut self, mut op: OperationIr) {
        if matches!(op, OperationIr::Drop(_)) { return; }
        self.assert_open();
        op.visit_mut(&mut AliasVisitor { aliases: &self.aliases });
        self.operations.push(op);
    }
}

struct AliasVisitor<'a> { aliases: &'a BTreeMap<TensorId, TensorId> }
impl IrVisitorMut for AliasVisitor<'_> {
    fn visit_tensor_mut(&mut self, tensor: &mut TensorIr) {
        while let Some(source) = self.aliases.get(&tensor.id) { tensor.id = *source; }
    }
}

#[derive(Clone)]
pub struct CaptureClient { device: CaptureDevice, session: Arc<CaptureSession> }

impl RunnerClient for CaptureClient {
    type Device = CaptureDevice;
    fn context_id(&self) -> Option<usize> { Some(Arc::as_ptr(&self.session) as usize) }
    fn register_op(&self, op: OperationIr) { self.session.state.lock().register_op(op); }
    fn read_tensor_async(&self, tensor: TensorIr) -> DynFut<Result<TensorData, ExecutionError>> {
        let value = self.session.state.lock().value(tensor.id);
        Box::pin(async move { value.ok_or_else(|| ExecutionError::WithContext {
            reason: format!("Captured tensor {} has no concrete value", tensor.id),
        }) })
    }
    fn sync(&self) -> Result<(), ExecutionError> { Ok(()) }
    fn flush(&self) -> Result<(), ExecutionError> { Ok(()) }
    fn create_empty_handle(&self) -> TensorId {
        self.session.state.lock().assert_open();
        TensorId::new(TENSOR_COUNTER.fetch_add(1, Ordering::Relaxed))
    }
    fn register_tensor_data(&self, data: TensorData) -> RouterTensor<Self> {
        let mut state = self.session.state.lock();
        state.assert_open();
        assert!(!matches!(data.dtype, DType::QFloat(_)), "Quantized capture tensors are not supported");
        let id = TensorId::new(TENSOR_COUNTER.fetch_add(1, Ordering::Relaxed));
        let shape = data.shape.clone();
        let dtype = data.dtype;
        state.values.insert(id, data);
        drop(state);
        RouterTensor::new(id, shape, dtype, self.clone())
    }
    fn device(&self) -> CaptureDevice { self.device }
    fn seed(&self, _: u64) { panic!("Seeding is not supported during graph capture"); }
    fn dtype_usage(&self, dtype: DType) -> DTypeUsageSet {
        if matches!(dtype, DType::QFloat(_)) { DTypeUsageSet::empty() } else { DTypeUsage::general() }
    }
    fn register_graph(&self, graph: GraphIr) -> Result<GraphId, ExecutionError> {
        let graph = Arc::new(CachedGraph::new(graph)?);
        let mut state = self.session.state.lock();
        state.assert_open();
        let id = GraphId(GRAPH_COUNTER.fetch_add(1, Ordering::Relaxed));
        state.graphs.insert(id, graph);
        Ok(id)
    }
    fn execute_graph(&self, id: GraphId, bindings: GraphBindings) -> Result<(), ExecutionError> {
        let mut state = self.session.state.lock();
        state.assert_open();
        let graph = state.graphs.get(&id).cloned().ok_or_else(|| ExecutionError::WithContext {
            reason: format!("Capture graph {id:?} was not registered"),
        })?;
        graph.replay(bindings, |reserved| {
            loop {
                let id = TensorId::new(TENSOR_COUNTER.fetch_add(1, Ordering::Relaxed));
                if !reserved.contains(&id) { return Ok(id); }
            }
        }, |op| state.register_op(op))
    }
    fn remove_graph(&self, id: GraphId) -> Result<(), ExecutionError> {
        self.session.state.lock().graphs.remove(&id);
        Ok(())
    }
    fn register_alias(&self, destination: TensorId, source: TensorId) -> Result<(), ExecutionError> {
        let mut state = self.session.state.lock();
        state.assert_open();
        let source = state.resolve_alias(source);
        if source != destination { state.aliases.insert(destination, source); }
        Ok(())
    }
}
