use std::{collections::HashSet, marker::PhantomData, sync::atomic::Ordering};
use ruda_fusion::{FuserProperties, FuserStatus, FusionBackend, FusionRuntime, NumOperations,
    OperationFuser, OperationRan, Optimization, stream::{Context, Operation, OrderedExecution}};
use ruda_tensor::{DType, ops::FloatTensorOps, graph::{BackendIr, CustomOpIr, GraphBindings,
    GraphIr, Handle, HandleContainer, OperationIr, ScalarIr, TensorHandle, TensorId,
    TensorIr, TensorStatus}};
use serde::{Deserialize, Serialize};
use crate::{BackendRouter, QueuedGraphId, RouterTensor, RunnerChannel, RunnerClient, get_client};

impl<R: RunnerChannel> BackendIr for BackendRouter<R> {
    type Handle = RouterTensor<R::Client>;
    fn float_tensor(handle: TensorHandle<Self::Handle>) -> Self::FloatTensorPrimitive { handle.handle }
    fn int_tensor(handle: TensorHandle<Self::Handle>) -> Self::IntTensorPrimitive { handle.handle }
    fn bool_tensor(handle: TensorHandle<Self::Handle>) -> Self::BoolTensorPrimitive { handle.handle }
    fn quantized_tensor(handle: TensorHandle<Self::Handle>) -> Self::QuantizedTensorPrimitive { handle.handle }
    fn float_tensor_handle(tensor: Self::FloatTensorPrimitive) -> Self::Handle { tensor }
    fn int_tensor_handle(tensor: Self::IntTensorPrimitive) -> Self::Handle { tensor }
    fn bool_tensor_handle(tensor: Self::BoolTensorPrimitive) -> Self::Handle { tensor }
    fn quantized_tensor_handle(tensor: Self::QuantizedTensorPrimitive) -> Self::Handle { tensor }
}

impl<R: RunnerChannel> FusionBackend for BackendRouter<R> {
    type FusionRuntime = RouterFusionRuntime<R>;
    type FullPrecisionBackend = Self;
    fn cast_float(tensor: Self::FloatTensorPrimitive, dtype: DType) -> Self::Handle {
        Self::float_cast(tensor, dtype.into())
    }
}

/// Fusion runtime that caches router operation graphs rather than device kernels.
pub struct RouterFusionRuntime<R: RunnerChannel>(PhantomData<R>);

impl<R: RunnerChannel> core::fmt::Debug for RouterFusionRuntime<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { f.write_str("RouterFusionRuntime") }
}

impl<R: RunnerChannel> FusionRuntime for RouterFusionRuntime<R> {
    type OptimizationState = RouterGraphExecutionState;
    type Optimization = RouterGraphExecution<R>;
    type FusionHandle = RouterTensor<R::Client>;
    type FusionDevice = R::Device;

    fn fusers(device: R::Device) -> Vec<Box<dyn OperationFuser<Self::Optimization>>> {
        vec![Box::new(RouterFuser::<R>::new(device))]
    }

    fn alias_handle(handle: &Self::FusionHandle) -> Self::FusionHandle {
        let id = handle.client.create_empty_handle();
        handle.client.queue_alias(id, handle.id).expect("Router tensor alias submission failed");
        RouterTensor::new(id, handle.shape.clone(), handle.dtype, handle.client.clone())
    }

    fn free_handle(handles: &mut HandleContainer<Self::FusionHandle>, tensor: &TensorIr, ran: OperationRan) {
        if tensor.status != TensorStatus::ReadWrite { return; }
        if ran == OperationRan::No { handles.free(tensor); return; }
        if let Some(Handle::Existing(handle)) = handles.remove_handle(tensor.id) {
            handle.count.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Greedy graph fuser using replay-byte savings and configurable growth limits.
pub struct RouterFuser<R: RunnerChannel> {
    device: R::Device,
    operations: Vec<OperationIr>,
    closed_on_init: bool,
    savings: ReplaySavings,
    score: u64,
    score_max: u64,
    num_since_max_unchanged: usize,
    max_graph_size: Option<usize>,
    growth_patience: usize,
}

impl<R: RunnerChannel> RouterFuser<R> {
    fn new(device: R::Device) -> Self {
        let config = ruda_tensor_config::config();
        let fusion = &config.fusion;
        Self { device, operations: Vec::new(), closed_on_init: false, savings: ReplaySavings::default(),
            score: 0, score_max: 0, num_since_max_unchanged: 0,
            max_graph_size: fusion.max_graph_size, growth_patience: fusion.growth_patience }
    }
}

impl<R: RunnerChannel> Clone for RouterFuser<R> {
    fn clone(&self) -> Self {
        Self { device: self.device.clone(), operations: self.operations.clone(), closed_on_init: self.closed_on_init,
            savings: self.savings.clone(), score: self.score, score_max: self.score_max,
            num_since_max_unchanged: self.num_since_max_unchanged,
            max_graph_size: self.max_graph_size, growth_patience: self.growth_patience }
    }
}

impl<R: RunnerChannel> OperationFuser<RouterGraphExecution<R>> for RouterFuser<R> {
    fn fuse(&mut self, operation: &OperationIr) {
        if self.closed_on_init { return; }
        if matches!(operation, OperationIr::Init(_)) { self.closed_on_init = true; return; }
        self.savings.add(operation);
        self.operations.push(operation.clone());
        self.score = self.savings.percent() * 100 + 1;
        if self.score > self.score_max {
            self.score_max = self.score;
            self.num_since_max_unchanged = 0;
        } else { self.num_since_max_unchanged += 1; }
    }
    fn finish(&mut self) -> RouterGraphExecution<R> {
        let operations = core::mem::take(&mut self.operations);
        self.reset();
        RouterGraphExecution::new(operations, self.device.clone())
    }
    fn reset(&mut self) {
        self.operations.clear();
        self.closed_on_init = false;
        self.savings = ReplaySavings::default();
        self.score = 0;
        self.score_max = 0;
        self.num_since_max_unchanged = 0;
    }
    fn status(&self) -> FuserStatus {
        if self.closed_on_init || self.num_since_max_unchanged >= self.growth_patience
            || self.max_graph_size.is_some_and(|max| self.operations.len() > max) {
            FuserStatus::Closed
        } else { FuserStatus::Open }
    }
    fn properties(&self) -> FuserProperties {
        FuserProperties { score: self.score, ready: self.operations.len() > 1 }
    }
    fn len(&self) -> usize { self.operations.len() }
    fn clone_dyn(&self) -> Box<dyn OperationFuser<RouterGraphExecution<R>>> { Box::new(self.clone()) }
}

/// Unfused handler for custom operations understood by the selected runner client.
pub struct CustomOperation<R: RunnerChannel> { ir: CustomOpIr, device: R::Device }

impl<R: RunnerChannel> core::fmt::Debug for CustomOperation<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CustomOperation").field("id", &self.ir.id).finish()
    }
}

impl<R: RunnerChannel> CustomOperation<R> {
    /// Construct a handler with concrete operation metadata and device.
    pub fn new(ir: CustomOpIr, device: R::Device) -> Self { Self { ir, device } }
}

impl<R: RunnerChannel> Operation<RouterFusionRuntime<R>> for CustomOperation<R> {
    fn execute(&self, handles: &mut HandleContainer<RouterTensor<R::Client>>) {
        let client = self.ir.inputs.first().and_then(|input| handles.get_handle_ref(&input.id))
            .map(|handle| handle.client.clone()).unwrap_or_else(|| get_client::<R>(&self.device));
        let inputs = self.ir.inputs.iter().map(|input|
            handles.get_handle(&input.id, &input.status).into_ir()).collect();
        let outputs: Vec<_> = self.ir.outputs.iter().map(|out|
            RouterTensor::new(client.create_empty_handle(), out.shape.clone(), out.dtype, client.clone())).collect();
        client.register_op(OperationIr::Custom(CustomOpIr {
            id: self.ir.id.clone(), inputs, outputs: outputs.iter().map(|out| out.to_ir_out()).collect(),
            scalars: self.ir.scalars.clone(),
        }));
        for (out, tensor) in self.ir.outputs.iter().zip(outputs) { handles.register_handle(out.id, tensor); }
    }
}

/// Relative graph registered once, with only boundary bindings sent on subsequent executions.
pub struct RouterGraphExecution<R: RunnerChannel> {
    graph: GraphIr,
    consumed_inputs: HashSet<TensorId>,
    device: R::Device,
    registration: Option<(R::Client, QueuedGraphId)>,
}

/// Portable graph state; runner-local registration IDs are not persisted.
#[derive(Serialize, Deserialize)]
pub struct RouterGraphExecutionState { operations: Vec<OperationIr> }

impl<R: RunnerChannel> RouterGraphExecution<R> {
    fn new(operations: Vec<OperationIr>, device: R::Device) -> Self {
        let (inputs, outputs) = GraphIr::classify(&operations);
        let mut consumed: HashSet<_> = operations.iter().flat_map(|op| op.inputs().chain(op.outputs()))
            .filter(|tensor| tensor.status == TensorStatus::ReadWrite).map(|tensor| tensor.id).collect();
        for operation in &operations {
            if let OperationIr::Drop(tensor) = operation { consumed.insert(tensor.id); }
        }
        let consumed_inputs = inputs.iter().copied().filter(|id| consumed.contains(id)).collect();
        Self { graph: GraphIr { operations, inputs, outputs }, consumed_inputs, device, registration: None }
    }
}

impl<R: RunnerChannel> core::fmt::Debug for RouterGraphExecution<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RouterGraphExecution").field("len", &self.graph.len()).finish()
    }
}

impl<R: RunnerChannel> NumOperations for RouterGraphExecution<R> {
    fn len(&self) -> usize { self.graph.len() }
    fn name(&self) -> &'static str { "RouterGraphExecution" }
}

impl<R: RunnerChannel> Optimization<RouterFusionRuntime<R>> for RouterGraphExecution<R> {
    fn execute(&mut self, context: &mut Context<RouterTensor<R::Client>>,
        _execution: &OrderedExecution<RouterFusionRuntime<R>>) {
        let client = self.graph.inputs.first().and_then(|input| context.tensors.get(input))
            .and_then(|global| context.handles.get_handle_ref(&global.id))
            .map(|handle| handle.client.clone()).unwrap_or_else(|| get_client::<R>(&self.device));
        let reusable = self.registration.as_ref().is_some_and(|(owner, _)|
            owner.context_id().is_some() && owner.context_id() == client.context_id());
        if !reusable {
            if let Some((owner, id)) = self.registration.take() {
                if let Err(error) = owner.queue_graph_release(id) {
                    log::warn!("Router graph release failed: {error:?}");
                }
            }
        }
        let mut tensors = Vec::with_capacity(self.graph.inputs.len() + self.graph.outputs.len());
        let mut consumed = Vec::new();
        for &input in &self.graph.inputs {
            let global = context.tensors.get(&input).expect("Missing router graph input binding");
            if self.consumed_inputs.contains(&input) {
                let handle = context.handles.get_handle(&global.id, &TensorStatus::ReadWrite);
                let handle = if handle.count.load(Ordering::Acquire) > 1 {
                    RouterFusionRuntime::<R>::alias_handle(&handle)
                } else { handle };
                tensors.push((input, handle.id));
                consumed.push(handle);
            } else {
                let handle = context.handles.get_handle_ref(&global.id).expect("Missing router graph input handle");
                tensors.push((input, handle.id));
            }
        }
        for &output in &self.graph.outputs {
            let global = context.tensors.get(&output).expect("Missing router graph output binding");
            let id = client.create_empty_handle();
            tensors.push((output, id));
            context.handles.register_handle(global.id,
                RouterTensor::new(id, global.shape.clone(), global.dtype, client.clone()));
        }
        let mut shapes = vec![0usize; context.shapes_relative2global.len()];
        for (relative, concrete) in &context.shapes_relative2global { shapes[*relative] = *concrete; }
        let mut scalars = vec![ScalarIr::UInt(0); context.scalars.len()];
        for (id, value) in &context.scalars { scalars[id.value as usize] = *value; }
        let bindings = GraphBindings { tensors, shapes, scalars, ranges: context.ranges.clone() };
        match &self.registration {
            Some((_, id)) => client.queue_graph_replay(*id, bindings).expect("Router graph replay submission failed"),
            None => {
                let id = client.queue_graph(self.graph.clone(), bindings).expect("Router graph submission failed");
                self.registration = Some((client.clone(), id));
            }
        }
        for handle in consumed { handle.count.fetch_add(1, Ordering::Relaxed); }
    }
    fn to_state(&self) -> RouterGraphExecutionState {
        RouterGraphExecutionState { operations: self.graph.operations.clone() }
    }
    fn from_state(device: &R::Device, state: RouterGraphExecutionState) -> Self {
        Self::new(state.operations, device.clone())
    }
}

impl<R: RunnerChannel> Drop for RouterGraphExecution<R> {
    fn drop(&mut self) {
        if let Some((client, id)) = self.registration.take() {
            if let Err(error) = client.queue_graph_release(id) {
                log::warn!("Router graph release failed: {error:?}");
            }
        }
    }
}

#[derive(Clone, Default)]
struct ReplaySavings {
    baseline: u64,
    dimensions: HashSet<usize>,
    referenced: HashSet<TensorId>,
    produced: HashSet<TensorId>,
    consumed: HashSet<TensorId>,
    inputs: usize,
    outputs: usize,
}

impl ReplaySavings {
    fn add(&mut self, operation: &OperationIr) {
        let nodes = operation.nodes();
        self.baseline += 8;
        for tensor in nodes.iter().copied().chain(operation.outputs()) {
            self.baseline += 10;
            for &dimension in tensor.shape.iter() { self.baseline += 8; self.dimensions.insert(dimension); }
        }
        if let OperationIr::Drop(tensor) = operation { self.consume(tensor.id); }
        if !matches!(operation, OperationIr::Init(_)) {
            for tensor in operation.outputs() {
                if self.produced.insert(tensor.id) {
                    if self.referenced.contains(&tensor.id) { self.inputs -= 1; }
                    if !self.consumed.contains(&tensor.id) { self.outputs += 1; }
                }
            }
        }
        for tensor in nodes {
            if self.referenced.insert(tensor.id) && !self.produced.contains(&tensor.id) { self.inputs += 1; }
            if tensor.status == TensorStatus::ReadWrite { self.consume(tensor.id); }
        }
    }
    fn consume(&mut self, id: TensorId) {
        if self.consumed.insert(id) && self.produced.contains(&id) { self.outputs -= 1; }
    }
    fn percent(&self) -> u64 {
        if self.baseline == 0 { return 0; }
        let bindings = (self.inputs + self.outputs) as u64 * 16 + self.dimensions.len() as u64 * 8;
        (self.baseline.saturating_sub(bindings) * 100 / self.baseline).min(100)
    }
}
