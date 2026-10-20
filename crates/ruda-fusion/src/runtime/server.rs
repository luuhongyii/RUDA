use std::sync::Arc;

use crate::{
    FusionBackend, FusionRuntime, UnfusedOp,
    stream::{MultiStream, OperationStreams, StreamId},
};
use ruda_tensor::{TensorData, backend::ExecutionError};
use ruda_tensor::graph::{HandleContainer, OperationIr, TensorId, TensorIr};
use ruda_core::{device::CommunicationId, stub::RwLock};
use hashbrown::{HashMap, HashSet};

pub(crate) struct FusionUtilities {
    // Used in client using a downcast.
    #[allow(dead_code)]
    pub(crate) initialized_comms: RwLock<HashSet<CommunicationId>>,
}

pub struct FusionServer<R: FusionRuntime> {
    streams: MultiStream<R>,
    pub(crate) handles: HandleContainer<R::FusionHandle>,
    pub(crate) utilities: Arc<FusionUtilities>,
    pub(super) profile_windows: HashMap<u64, super::profile::Window>,
    pub(super) profile_counter: u64,
}

impl<R> FusionServer<R>
where
    R: FusionRuntime,
{
    pub fn new(device: R::FusionDevice, utilities: FusionUtilities) -> Self {
        Self {
            streams: MultiStream::new(device.clone()),
            handles: HandleContainer::new(),
            utilities: Arc::new(utilities),
            profile_windows: HashMap::new(),
            profile_counter: 0,
        }
    }

    pub fn register(
        &mut self,
        streams: OperationStreams,
        repr: OperationIr,
        operation: UnfusedOp<R>,
    ) {
        self.streams
            .register(streams, repr, operation, &mut self.handles)
    }

    pub fn drain_stream(&mut self, id: StreamId) {
        self.streams.drain(&mut self.handles, id)
    }

    pub(crate) fn discard_stream(&mut self, id: StreamId) {
        self.streams.discard(&mut self.handles, id)
    }

    pub(crate) fn shutdown(&mut self) {
        self.profile_windows.clear();
        self.streams.drain_all(&mut self.handles);
    }

    pub(crate) fn tag_shared_view(&mut self, stream: StreamId, source: TensorId, destination: TensorId) {
        if self.handles.get_handle_ref(&source).is_none() {
            self.drain_stream(stream);
        }
        let handle = self.handles.get_handle_ref(&source).expect("Shared tensor is materialized");
        self.handles.register_handle(destination, R::alias_handle(handle));
    }

    pub fn read_float<B>(&mut self, tensor: TensorIr, id: StreamId) -> B::FloatTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // Make sure all registered operations are executed.
        // The underlying backend can still be async.
        self.drain_stream(id);
        let tensor_float = self.handles.get_float_tensor::<B>(&tensor);
        self.streams.mark_read(id, &tensor, &self.handles);
        tensor_float
    }

    pub fn read_int<B>(&mut self, tensor: TensorIr, id: StreamId) -> B::IntTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // Make sure all registered operations are executed.
        // The underlying backend can still be async.
        self.drain_stream(id);
        let tensor_int = self.handles.get_int_tensor::<B>(&tensor);
        self.streams.mark_read(id, &tensor, &self.handles);
        tensor_int
    }

    pub fn read_bool<B>(&mut self, tensor: TensorIr, id: StreamId) -> B::BoolTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // Make sure all registered operations are executed.
        // The underlying backend can still be async.
        self.drain_stream(id);
        let tensor_bool = self.handles.get_bool_tensor::<B>(&tensor);
        self.streams.mark_read(id, &tensor, &self.handles);
        tensor_bool
    }

    pub fn read_quantized<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> B::QuantizedTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // Make sure all registered operations are executed.
        // The underlying backend can still be async.
        self.drain_stream(id);
        let tensor_q = self.handles.get_quantized_tensor::<B>(&tensor);
        self.streams.mark_read(id, &tensor, &self.handles);
        tensor_q
    }

    pub fn float_data<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send + use<R, B>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        B::float_into_data(self.read_float::<B>(tensor, id))
    }

    pub fn int_data<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send + use<R, B>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        B::int_into_data(self.read_int::<B>(tensor, id))
    }

    pub fn bool_data<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send + use<R, B>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        B::bool_into_data(self.read_bool::<B>(tensor, id))
    }

    pub fn quantized_data<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send + use<R, B>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        B::q_into_data(self.read_quantized::<B>(tensor, id))
    }

    pub fn resolve_server_float<B>(&mut self, tensor: &TensorIr) -> B::FloatTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.handles.get_float_tensor::<B>(tensor)
    }

    pub fn resolve_server_int<B>(&mut self, tensor: &TensorIr) -> B::IntTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.handles.get_int_tensor::<B>(tensor)
    }

    pub fn resolve_server_bool<B>(&mut self, tensor: &TensorIr) -> B::BoolTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.handles.get_bool_tensor::<B>(tensor)
    }
}
