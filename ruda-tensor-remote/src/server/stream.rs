use core::marker::PhantomData;
use std::sync::Arc;

use crate::shared::{ConnectionId, LocalTransferId, SessionId, TaskResponse, TensorRemote};

use super::local_comm::LocalCommService;
use super::processor::{Processor, ProcessorTask};
use ruda_tensor::{ProfileOptions, ProfileToken, TensorData};
use ruda_communication::{
    Protocol,
    data_service::{TensorDataService, TensorTransferId},
};
use ruda_tensor::graph::{BackendIr, GraphBindings, GraphId, GraphIr, OperationIr, TensorId, TensorIr};
use ruda_tensor_router::Runner;
use ruda_core::tensor::DType;
use tokio::sync::mpsc::{Receiver, Sender};

/// The processor or the responder of a stream has stopped; the session must be closed.
#[derive(Debug)]
pub struct StreamClosed;

/// A stream makes sure all operations registered are executed in the order they were sent to the
/// server, potentially waiting to reconstruct consistency.
#[derive(Clone)]
pub struct Stream<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    compute_sender: Sender<ProcessorTask>,
    writer_sender: Sender<Receiver<TaskResponse>>,
    worker: Arc<tokio::sync::Mutex<Option<tokio::task::JoinHandle<()>>>>,
    _p: PhantomData<B>,
    _n: PhantomData<P>,
}

impl<B, P> Stream<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    pub async fn new(
        runner: Runner<B>,
        writer_sender: Sender<Receiver<TaskResponse>>,
        data_service: Arc<TensorDataService<B, P>>,
    ) -> Self {
        Self::new_inner(runner, writer_sender, data_service, None, None).await
    }

    pub(crate) async fn new_with_local_comm(
        runner: Runner<B>,
        writer_sender: Sender<Receiver<TaskResponse>>,
        data_service: Arc<TensorDataService<B, P>>,
        session: SessionId,
        local_comm: Arc<LocalCommService<B>>,
        closed: tokio_util::sync::CancellationToken,
        probe: super::processor::WorkerProbe,
    ) -> Self {
        Self::new_inner(runner, writer_sender, data_service, Some((session, local_comm, closed)), Some(probe)).await
    }

    async fn new_inner(
        runner: Runner<B>,
        writer_sender: Sender<Receiver<TaskResponse>>,
        data_service: Arc<TensorDataService<B, P>>,
        local_comm: Option<(SessionId, Arc<LocalCommService<B>>, tokio_util::sync::CancellationToken)>,
        probe: Option<super::processor::WorkerProbe>,
    ) -> Self {
        let (sender, worker) = Processor::<B, P>::start(runner, data_service, local_comm, probe).await;

        Self {
            compute_sender: sender,
            writer_sender,
            worker: Arc::new(tokio::sync::Mutex::new(Some(worker))),
            _p: PhantomData,
            _n: PhantomData,
        }
    }

    async fn submit(&self, task: ProcessorTask) -> Result<(), StreamClosed> {
        self.compute_sender.send(task).await.map_err(|_| StreamClosed)
    }

    async fn submit_with_response(
        &self,
        task: impl FnOnce(Sender<TaskResponse>) -> ProcessorTask,
    ) -> Result<(), StreamClosed> {
        let (sender, receiver) = tokio::sync::mpsc::channel(1);
        self.submit(task(sender)).await?;
        self.writer_sender.send(receiver).await.map_err(|_| StreamClosed)
    }

    pub async fn register_operation(&self, op: Box<OperationIr>) -> Result<(), StreamClosed> {
        self.submit(ProcessorTask::RegisterOperation(op)).await
    }

    /// Register a relative graph and acknowledge its runner-issued ID.
    pub async fn register_graph(&self, id: ConnectionId, graph: GraphIr) -> Result<(), StreamClosed> {
        self.submit_with_response(|sender| ProcessorTask::RegisterGraph(id, graph, sender)).await
    }

    /// Dispatch a concrete replay; acknowledgement is not a device completion fence.
    pub async fn execute_graph(
        &self,
        id: ConnectionId,
        graph: GraphId,
        bindings: GraphBindings,
    ) -> Result<(), StreamClosed> {
        self.submit_with_response(|sender| ProcessorTask::ExecuteGraph(id, graph, bindings, sender))
            .await
    }

    /// Release a registration without canceling previously submitted work.
    pub async fn remove_graph(&self, id: ConnectionId, graph: GraphId) -> Result<(), StreamClosed> {
        self.submit_with_response(|sender| ProcessorTask::RemoveGraph(id, graph, sender)).await
    }

    pub async fn register_alias(&self, id: ConnectionId, destination: TensorId, source: TensorId) -> Result<(), StreamClosed> {
        self.submit_with_response(|sender| ProcessorTask::RegisterAlias(id, destination, source, sender)).await
    }

    pub(crate) async fn queue_graph_replay(&self, id: GraphId, graph: Arc<ruda_tensor_router::CachedGraph>, bindings: GraphBindings)
        -> Result<(), StreamClosed> {
        self.submit(ProcessorTask::QueueGraphReplay(id, graph, bindings)).await
    }

    pub(crate) async fn queue_alias(&self, destination: TensorId, source: TensorId) -> Result<(), StreamClosed> {
        self.submit(ProcessorTask::QueueAlias(destination, source)).await
    }

    pub async fn flush(&self, id: ConnectionId) -> Result<(), StreamClosed> {
        self.submit_with_response(|sender| ProcessorTask::Flush(id, sender)).await
    }

    pub async fn device_info(&self, id: ConnectionId, info: crate::RemoteDeviceInfo) -> Result<(), StreamClosed> {
        self.submit_with_response(|sender| ProcessorTask::DeviceInfo(id, info, sender)).await
    }

    pub async fn sync_collective(&self, id: ConnectionId) -> Result<(), StreamClosed> {
        self.submit_with_response(|sender| ProcessorTask::SyncCollective(id, sender)).await
    }

    pub async fn profile_start(&self, id: ConnectionId) -> Result<(), StreamClosed> {
        self.submit_with_response(|sender| ProcessorTask::ProfileStart(id, sender)).await
    }

    pub async fn profile_end(&self, id: ConnectionId, token: ProfileToken, options: ProfileOptions) -> Result<(), StreamClosed> {
        self.submit_with_response(|sender| ProcessorTask::ProfileEnd(id, token, options, sender)).await
    }

    pub async fn profile_abandon(&self, token: ProfileToken) -> Result<(), StreamClosed> {
        self.submit(ProcessorTask::ProfileAbandon(token)).await
    }

    pub async fn register_tensor(&self, tensor_id: TensorId, data: TensorData) -> Result<(), StreamClosed> {
        self.submit(ProcessorTask::RegisterTensor(tensor_id, data)).await
    }

    pub async fn register_tensor_remote(
        &self,
        tensor: TensorRemote,
        new_id: TensorId,
    ) -> Result<(), StreamClosed> {
        self.submit(ProcessorTask::RegisterTensorRemote(tensor, new_id)).await
    }

    pub(crate) async fn register_tensor_local(&self, transfer_id: LocalTransferId, new_id: TensorId)
        -> Result<(), StreamClosed> {
        self.submit(ProcessorTask::RegisterTensorLocal { transfer_id, new_id }).await
    }

    pub(crate) async fn expose_tensor_local(&self, tensor: TensorIr, transfer_id: LocalTransferId)
        -> Result<(), StreamClosed> {
        self.submit(ProcessorTask::ExposeTensorLocal { tensor, transfer_id }).await
    }

    pub async fn expose_tensor_remote(
        &self,
        tensor: TensorIr,
        count: u32,
        transfer_id: TensorTransferId,
    ) -> Result<(), StreamClosed> {
        self.submit(ProcessorTask::ExposeTensorRemote {
            tensor,
            count,
            transfer_id,
        })
        .await
    }

    pub async fn read_tensor(&self, id: ConnectionId, desc: TensorIr) -> Result<(), StreamClosed> {
        self.submit_with_response(|sender| ProcessorTask::ReadTensor(id, desc, sender)).await
    }

    pub(crate) async fn expose_tensor_capability(&self, tensor: TensorIr, count: u32,
        capability: ruda_communication::data_service::TransferCapability, target: ruda_communication::PeerId) -> Result<(), StreamClosed> {
        self.submit(ProcessorTask::ExposeTensorCapability { tensor, count, capability, target }).await
    }

    pub(crate) async fn register_tensor_capability(&self, address: ruda_communication::Address,
        capability: ruda_communication::data_service::TransferCapability, new_id: TensorId) -> Result<(), StreamClosed> {
        self.submit(ProcessorTask::RegisterTensorCapability { address, capability, new_id }).await
    }

    pub async fn sync(&self, id: ConnectionId) -> Result<(), StreamClosed> {
        self.submit_with_response(|sender| ProcessorTask::Sync(id, sender)).await
    }

    /// Ask the processor to drain and stop. A processor that already stopped needs no cleanup.
    pub async fn close(&self) {
        let mut worker = self.worker.lock().await;
        if let Some(worker) = worker.take() {
            let _ = self.submit(ProcessorTask::Close).await;
            if let Err(err) = worker.await {
                log::error!("Stream processor stopped with an error: {err}");
            }
        }
    }

    pub async fn seed(&self, seed: u64) -> Result<(), StreamClosed> {
        self.submit(ProcessorTask::Seed(seed)).await
    }

    pub async fn rng_state(&self, id: ConnectionId) -> Result<(), StreamClosed> {
        self.submit_with_response(|sender| ProcessorTask::RngState(id, sender)).await
    }

    pub async fn restore_rng_state(&self, id: ConnectionId, state: Vec<u8>) -> Result<(), StreamClosed> {
        self.submit_with_response(|sender| ProcessorTask::RestoreRngState(id, state, sender)).await
    }

    pub async fn dtype_usage(&self, id: ConnectionId, dtype: DType) -> Result<(), StreamClosed> {
        self.submit_with_response(|sender| ProcessorTask::DTypeUsage(id, dtype, sender)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruda_communication::websocket::WebSocket;
    use ruda_tensor_host::Host;
    use tokio_util::sync::CancellationToken;

    #[tokio::test(flavor = "multi_thread")]
    async fn close_waits_for_processor_and_is_idempotent() {
        let data_service = Arc::new(TensorDataService::new(CancellationToken::new()));
        let (writer, _reader) = tokio::sync::mpsc::channel(1);
        let stream = Stream::<Host, WebSocket>::new(
            Runner::new(Default::default()), writer, data_service,
        ).await;
        stream.seed(1).await.unwrap();
        stream.close().await;
        assert!(stream.compute_sender.is_closed());
        assert!(stream.worker.lock().await.is_none());
        stream.close().await;
        assert!(stream.seed(2).await.is_err());
    }
}
