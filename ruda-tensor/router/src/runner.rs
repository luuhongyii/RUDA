mod base_float;
mod base_quantized;
mod base_int;
mod base_bool;
mod numeric_float;
mod numeric_int;
mod boolean;
mod integer;
mod float;
mod module;

#[cfg(test)]
mod tests;

use core::sync::atomic::{AtomicU64, Ordering};

use super::{RouterTensor, RunnerClient};
use crate::{
    binary_bool_ops, binary_float_cmp_ops, binary_float_ops, binary_int_cmp_ops, binary_int_ops,
    reduce_float_dim_ops, reduce_float2int_dim_ops, reduce_int_dim_ops, scalar_float_cmp_ops,
    scalar_float_ops, scalar_int_cmp_ops, scalar_int_ops, unary_float_ops, unary_int_ops,
};
use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::sync::Arc;
use ruda_tensor::{Backend, DType, ExecutionError, Shape, TensorData, tensor::IndexingUpdateOp};
use ruda_tensor::graph::{
    BackendIr, BaseOperationIr, BoolOperationIr, FloatOperationIr, HandleContainer, HandleKind, IntOperationIr,
    GraphBindings, GraphId, GraphIr, ModuleOperationIr, NumericOperationIr, OperationIr, TensorId, TensorIr, TensorStatus,
};
use ruda_core::{future::DynFut, stub::Mutex};

/// A runner's context contains a [handle container](HandleContainer) to manage
/// (i.e., fetch and update) existing tensors.
pub struct RunnerContext<B: BackendIr> {
    /// Handle container to retrieve tensors based on their intermediate representation.
    handles: HandleContainer<B::Handle>,
    graphs: BTreeMap<GraphId, Arc<crate::graph::CachedGraph>>,
}

static COUNTER: AtomicU64 = AtomicU64::new(0);
static GRAPH_COUNTER: AtomicU64 = AtomicU64::new(0);
static GRAPH_TENSOR_COUNTER: AtomicU64 = AtomicU64::new(0);

impl<B: BackendIr> RunnerContext<B> {
    /// Create a new (uninitialized) empty tensor and returns its corresponding [tensor id](TensorId).
    fn create_empty_handle(&mut self) -> TensorId {
        let value = COUNTER.fetch_add(1, Ordering::Relaxed);
        TensorId::new(value)
    }
}

/// A runner is responsible for executing tensor operations for a given [intermediate backend](BackendIr).
#[derive(Clone)]
pub struct Runner<B: BackendIr> {
    // Mutex for the mutable handles
    context: Arc<Mutex<RunnerContext<B>>>,
    device: B::Device,
    custom_ops: crate::CustomOpRegistry<B>,
    #[cfg(feature = "distributed")]
    native_collectives: Option<NativeCollectives<B>>,
}

impl<B: BackendIr> core::fmt::Debug for Runner<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Runner")
            .field("device", &self.device)
            .finish()
    }
}

impl<B: BackendIr> Runner<B> {
    /// Create a new runner.
    pub fn new(device: B::Device) -> Self {
        Self::with_custom_ops(device, crate::CustomOpRegistry::new())
    }

    /// Create a runner with handlers for custom operations and registered graph replays.
    pub fn with_custom_ops(device: B::Device, custom_ops: crate::CustomOpRegistry<B>) -> Self {
        Self {
            context: Arc::new(Mutex::new(RunnerContext {
                handles: HandleContainer::new(),
                graphs: BTreeMap::new(),
            })),
            device,
            custom_ops,
            #[cfg(feature = "distributed")]
            native_collectives: None,
        }
    }

    /// Create a runner on the same device with its own, empty tensor and graph context.
    ///
    /// Unlike [`Clone`], which shares the context, the forked runner cannot observe or replace
    /// tensors registered through `self`, even when both use the same [`TensorId`] values.
    pub fn fork(&self) -> Self {
        let mut runner = self.clone();
        runner.context = Arc::new(Mutex::new(RunnerContext {
            handles: HandleContainer::new(),
            graphs: BTreeMap::new(),
        }));
        runner
    }

    /// Dispatch a validated graph snapshot, retaining it independently of registration lifetime.
    pub fn execute_cached_graph(&self, graph: &crate::CachedGraph, bindings: GraphBindings) -> Result<(), ExecutionError> {
        graph.replay(bindings, |reserved: &BTreeSet<TensorId>| {
            let context = self.context.lock().unwrap();
            loop {
                let value = GRAPH_TENSOR_COUNTER.fetch_update(Ordering::Relaxed, Ordering::Relaxed,
                    |value| value.checked_add(1).filter(|next| *next <= 1 << 63))
                    .map_err(|_| crate::graph::graph_error("graph intermediate ID space exhausted"))?;
                let tensor = TensorId::new(value | 1 << 63);
                if !reserved.contains(&tensor) && !context.handles.has_handle(&tensor) { return Ok(tensor); }
            }
        }, |operation| self.register_op(operation))
    }

    /// Create a runner forwarding collectives to a native distributed backend.
    /// Normal runners do not gain a hidden host-staged fallback.
    #[cfg(feature = "distributed")]
    pub fn new_distributed(device: B::Device) -> Self
    where B: ruda_tensor::distributed::DistributedBackend {
        Self::new(device).with_native_collectives()
    }

    /// Enable native collectives while preserving this runner's handlers and tensor context.
    #[cfg(feature = "distributed")]
    pub fn with_native_collectives(mut self) -> Self
    where B: ruda_tensor::distributed::DistributedBackend {
        fn enqueue<B: BackendIr + ruda_tensor::distributed::DistributedBackend>(
            tensor: ruda_tensor::tensor::FloatTensor<B>, op: ruda_tensor::distributed::ReduceOperation,
            devices: alloc::vec::Vec<ruda_tensor::DeviceId>) -> ruda_tensor::tensor::FloatTensor<B>
        {
            // The Router returns CollectiveTensor; its resolve() invokes native
            // sync_collective before the resulting float handle is exposed.
            unsafe { B::all_reduce(tensor, op, devices).assume_resolved() }
        }
        self.native_collectives = Some(NativeCollectives {
            enqueue: enqueue::<B>, synchronize: B::sync_collective,
        });
        self
    }

    #[cfg(feature = "distributed")]
    pub fn sync_native_collective(&self) {
        let native = self.native_collectives.as_ref()
            .expect("use NativeDistributedChannel or Runner::new_distributed");
        (native.synchronize)(&self.device);
    }

    /// Whether this interpreter was configured with native collective callbacks.
    pub fn has_native_collectives(&self) -> bool {
        #[cfg(feature = "distributed")]
        { self.native_collectives.is_some() }
        #[cfg(not(feature = "distributed"))]
        { false }
    }

    #[cfg(feature = "distributed")]
    fn run_distributed(&self, handles: &mut HandleContainer<B::Handle>,
        op: &ruda_tensor::graph::DistributedOperationIr)
    {
        use ruda_tensor::{DeviceOps, TensorMetadata};
        let native = self.native_collectives.as_ref()
            .expect("distributed IR requires NativeDistributedChannel or Runner::new_distributed");
        match op {
            ruda_tensor::graph::DistributedOperationIr::AllReduce(desc) => {
                let devices = desc.device_ids.iter().map(|&(kind, index)| ruda_tensor::DeviceId::new(kind,index))
                    .collect::<alloc::vec::Vec<_>>();
                let mut unique = devices.clone(); unique.sort(); unique.dedup();
                assert!(!devices.is_empty() && unique.len() == devices.len(), "invalid collective group");
                assert!(devices.contains(&self.device.id()), "runner device is not a collective participant");
                assert_eq!(desc.tensor.shape, desc.out.shape, "collective output shape mismatch");
                assert_eq!(desc.tensor.dtype, desc.out.dtype, "collective output dtype mismatch");
                let tensor = handles.get_float_tensor::<B>(&desc.tensor);
                assert_eq!(tensor.shape(), desc.tensor.shape, "collective input metadata mismatch");
                assert_eq!(tensor.dtype(), desc.tensor.dtype, "collective input dtype mismatch");
                let output = (native.enqueue)(tensor, desc.op, devices);
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
        }
    }

    /// Get the tensor handle for the given [tensor representation](TensorIr).
    pub fn get_tensor_handle(&self, tensor: &TensorIr) -> B::Handle {
        let handles = &mut self.context.lock().unwrap().handles;
        handles.get_tensor_handle(tensor).handle
    }

    /// Take or clone a device-resident primitive according to the descriptor's ownership.
    pub fn get_tensor(&self, tensor: &TensorIr) -> HandleKind<B> {
        let handles = &mut self.context.lock().unwrap().handles;
        match tensor.dtype {
            DType::QFloat(_) => HandleKind::Quantized(handles.get_quantized_tensor::<B>(tensor)),
            dtype if dtype.is_float() => HandleKind::Float(handles.get_float_tensor::<B>(tensor)),
            dtype if dtype.is_int() || dtype.is_uint() => HandleKind::Int(handles.get_int_tensor::<B>(tensor)),
            dtype if dtype.is_bool() => HandleKind::Bool(handles.get_bool_tensor::<B>(tensor)),
            dtype => panic!("Unsupported native tensor kind: {dtype:?}"),
        }
    }

    /// Move a native primitive onto this runner's device without serializing its data.
    pub fn register_tensor_to_device(&self, id: TensorId, tensor: HandleKind<B>) {
        let handle = match tensor {
            HandleKind::Float(tensor) => B::float_tensor_handle(B::float_to_device(tensor, &self.device)),
            HandleKind::Int(tensor) => B::int_tensor_handle(B::int_to_device(tensor, &self.device)),
            HandleKind::Bool(tensor) => B::bool_tensor_handle(B::bool_to_device(tensor, &self.device)),
            HandleKind::Quantized(tensor) => B::quantized_tensor_handle(B::q_to_device(tensor, &self.device)),
        };
        self.context.lock().unwrap().handles.register_handle(id, handle);
    }

    /// Create a tensor with the given handle and shape.
    pub fn register_tensor<C: RunnerClient>(
        &self,
        handle: B::Handle,
        shape: Shape,
        dtype: DType,
        client: C,
    ) -> RouterTensor<C> {
        let mut ctx = self.context.lock().unwrap();
        let id = ctx.create_empty_handle();

        ctx.handles.register_handle(id, handle);
        core::mem::drop(ctx);

        RouterTensor::new(id, shape, dtype, client)
    }

    /// Register a tensor from its data and id.
    pub fn register_tensor_data_id(&self, id: TensorId, data: TensorData) {
        let mut ctx = self.context.lock().unwrap();
        let dtype = data.dtype;

        if dtype.is_float() {
            let tensor = B::float_from_data(data, &self.device);
            ctx.handles.register_float_tensor::<B>(&id, tensor)
        } else if dtype.is_int() || dtype.is_uint() {
            let tensor = B::int_from_data(data, &self.device);
            ctx.handles.register_int_tensor::<B>(&id, tensor)
        } else if dtype.is_bool() {
            let tensor = B::bool_from_data(data, &self.device);
            ctx.handles.register_bool_tensor::<B>(&id, tensor)
        } else if let DType::QFloat(_) = dtype {
            let tensor = B::q_from_data(data, &self.device);
            ctx.handles.register_quantized_tensor::<B>(&id, tensor)
        }

        core::mem::drop(ctx);
    }

    /// Register a tensor and returns its intermediate representation.
    pub fn register_tensor_data_desc(&self, data: TensorData) -> TensorIr {
        let mut ctx = self.context.lock().unwrap();
        let id = ctx.create_empty_handle();
        let shape = data.shape.clone();
        let dtype = data.dtype;

        if dtype.is_float() {
            let tensor = B::float_from_data(data, &self.device);
            ctx.handles.register_float_tensor::<B>(&id, tensor)
        } else if dtype.is_int() || dtype.is_uint() {
            let tensor = B::int_from_data(data, &self.device);
            ctx.handles.register_int_tensor::<B>(&id, tensor)
        } else if dtype.is_bool() {
            let tensor = B::bool_from_data(data, &self.device);
            ctx.handles.register_bool_tensor::<B>(&id, tensor)
        } else if let DType::QFloat(_) = dtype {
            let tensor = B::q_from_data(data, &self.device);
            ctx.handles.register_quantized_tensor::<B>(&id, tensor)
        }

        core::mem::drop(ctx);

        TensorIr {
            id,
            shape,
            status: TensorStatus::ReadWrite,
            dtype,
        }
    }
}

// This is a Remote Runner
impl<B: BackendIr> RunnerClient for Runner<B> {
    fn context_id(&self) -> Option<usize> { Some(Arc::as_ptr(&self.context) as usize) }

    type Device = B::Device;

    fn avg_pool3d_native_output_size(&self, input: [usize; 3], kernel: [usize; 3],
        stride: [usize; 3], padding: [usize; 3], ceil: bool) -> Option<[usize; 3]> {
        B::avg_pool3d_native_output_size(input, kernel, stride, padding, ceil)
    }

    /// Execute a tensor operation.
    fn register_op(&self, op: OperationIr) {
        // Remove unused tensor handles
        let mut ctx = self.context.lock().unwrap();

        let handles = &mut ctx.handles;
        match &op {
            // For every op: get the input(s), execute the operation and register the output(s)
            OperationIr::BaseFloat(op) => self.run_base_float(handles, op),
            OperationIr::BaseInt(op) => self.run_base_int(handles, op),
            OperationIr::BaseBool(op) => self.run_base_bool(handles, op),
            OperationIr::NumericFloat(_dtype, op) => self.run_numeric_float(handles, op),
            OperationIr::NumericInt(_dtype, op) => self.run_numeric_int(handles, op),
            OperationIr::Bool(op) => self.run_boolean(handles, op),
            OperationIr::Int(op) => self.run_integer(handles, op),
            OperationIr::Float(_dtype, op) => self.run_float(handles, op),
            OperationIr::Module(op) => self.run_module(handles, op),
            OperationIr::Custom(desc) => match self.custom_ops.get(&desc.id) {
                Some(handler) => handler(handles, desc, &self.device),
                None => panic!("No custom-op handler registered for `{}`; use Runner::with_custom_ops", desc.id),
            },
            OperationIr::Init(_) => {
                // Nothing to do.
            }
            OperationIr::Drop(repr) => {
                handles.remove_handle(repr.id);
            }
            #[cfg(feature = "distributed")]
            OperationIr::Distributed(op) => self.run_distributed(handles, op),
        }
    }

    fn register_alias(&self, destination: TensorId, source: TensorId) -> Result<(), ExecutionError> {
        let mut context = self.context.lock().unwrap();
        if context.handles.has_handle(&destination) {
            return Err(crate::graph::graph_error("alias destination is already registered"));
        }
        let handle = context.handles.get_handle_ref(&source)
            .ok_or_else(|| crate::graph::graph_error("alias source is not initialized"))?.clone();
        context.handles.register_handle(destination, handle);
        Ok(())
    }

    fn register_graph(&self, graph: GraphIr) -> Result<GraphId, ExecutionError> {
        let graph = Arc::new(crate::graph::CachedGraph::new(graph)?);
        let id = GRAPH_COUNTER.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| value.checked_add(1))
            .map_err(|_| crate::graph::graph_error("graph ID space exhausted"))?;
        let id = GraphId(id);
        self.context.lock().unwrap().graphs.insert(id, graph);
        Ok(id)
    }

    fn execute_graph(&self, id: GraphId, bindings: GraphBindings) -> Result<(), ExecutionError> {
        let graph = self.context.lock().unwrap().graphs.get(&id).cloned()
            .ok_or_else(|| crate::graph::graph_error(alloc::format!("unknown registered graph {}", id.0)))?;
        self.execute_cached_graph(&graph, bindings)
    }

    fn remove_graph(&self, id: GraphId) -> Result<(), ExecutionError> {
        self.context.lock().unwrap().graphs.remove(&id)
            .ok_or_else(|| crate::graph::graph_error(alloc::format!("unknown registered graph {}", id.0)))?;
        Ok(())
    }

    fn read_tensor_async(&self, tensor: TensorIr) -> DynFut<Result<TensorData, ExecutionError>> {
        let mut ctx = self.context.lock().unwrap();

        enum Output<B: Backend> {
            Float(B::FloatTensorPrimitive),
            Int(B::IntTensorPrimitive),
            Bool(B::BoolTensorPrimitive),
            Quantized(B::QuantizedTensorPrimitive),
        }

        let tensor = if tensor.dtype.is_float() {
            let tensor = ctx.handles.get_float_tensor::<B>(&tensor);
            Output::<B>::Float(tensor)
        } else if tensor.dtype.is_int() || tensor.dtype.is_uint() {
            let tensor = ctx.handles.get_int_tensor::<B>(&tensor);
            Output::Int(tensor)
        } else if tensor.dtype.is_bool() {
            let tensor = ctx.handles.get_bool_tensor::<B>(&tensor);
            Output::Bool(tensor)
        } else if let DType::QFloat(_) = tensor.dtype {
            Output::Quantized(ctx.handles.get_quantized_tensor::<B>(&tensor))
        } else {
            unimplemented!()
        };

        match tensor {
            Output::Float(val) => Box::pin(B::float_into_data(val)),
            Output::Int(val) => Box::pin(B::int_into_data(val)),
            Output::Bool(val) => Box::pin(B::bool_into_data(val)),
            Output::Quantized(val) => Box::pin(B::q_into_data(val)),
        }
    }

    fn register_tensor_data(&self, data: TensorData) -> RouterTensor<Self> {
        let desc = self.register_tensor_data_desc(data);
        RouterTensor::new(desc.id, desc.shape, desc.dtype, self.clone())
    }

    fn device(&self) -> Self::Device {
        self.device.clone()
    }

    fn sync(&self) -> Result<(), ExecutionError> {
        #[cfg(feature = "distributed")]
        if let Some(native) = &self.native_collectives { (native.synchronize)(&self.device); }
        B::sync(&self.device)
    }

    fn rng_state(&self) -> Result<alloc::vec::Vec<u8>, ExecutionError> {
        self.sync()?;
        B::rng_state(&self.device)
    }
    fn restore_rng_state(&self, state: &[u8]) -> Result<(), ExecutionError> {
        self.sync()?;
        B::restore_rng_state(&self.device, state)
    }

    fn flush(&self) -> Result<(), ExecutionError> {
        B::flush(&self.device)
    }

    fn profile_start(&self) -> Result<Option<ruda_tensor::ProfileToken>, ExecutionError> {
        B::profile_start(&self.device)
    }
    fn profile_end(&self, token: ruda_tensor::ProfileToken, options: ruda_tensor::ProfileOptions)
        -> Result<ruda_tensor::ProfileDuration, ExecutionError> { B::profile_end(&self.device, token, options) }
    fn profile_abandon(&self, token: ruda_tensor::ProfileToken) { B::profile_abandon(&self.device, token); }

    fn graph_prepare(&self) -> Result<(), ExecutionError> {
        B::graph_prepare(&self.device)
    }

    fn graph_start_capture(&self) -> Result<(), ExecutionError> {
        B::graph_start_capture(&self.device)
    }

    fn graph_stop_capture(&self) -> Result<ruda_tensor::BackendGraph, ExecutionError> {
        B::graph_stop_capture(&self.device)
    }

    fn graph_abort_capture(&self) {
        B::graph_abort_capture(&self.device)
    }

    unsafe fn graph_replay(&self, graph: &ruda_tensor::BackendGraph) -> Result<(), ExecutionError> {
        unsafe { B::graph_replay(&self.device, graph) }
    }

    fn seed(&self, seed: u64) {
        B::seed(&self.device, seed)
    }

    fn create_empty_handle(&self) -> TensorId {
        let mut ctx = self.context.lock().unwrap();
        ctx.create_empty_handle()
    }

    fn dtype_usage(&self, dtype: DType) -> ruda_tensor::DTypeUsageSet {
        B::dtype_usage(&self.device, dtype)
    }
}


#[cfg(feature = "distributed")]
#[derive(Clone)]
struct NativeCollectives<B: BackendIr> {
    enqueue: fn(ruda_tensor::tensor::FloatTensor<B>, ruda_tensor::distributed::ReduceOperation,
        alloc::vec::Vec<ruda_tensor::DeviceId>) -> ruda_tensor::tensor::FloatTensor<B>,
    synchronize: fn(&B::Device),
}
