use core::marker::PhantomData;
use std::sync::Arc;

use crate::shared::{ConnectionId, TaskResponse, TensorRemote};

use super::processor::{Processor, ProcessorTask};
use ruda_tensor::TensorData;
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
        let (sender, worker) = Processor::<B, P>::start(runner, data_service).await;

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
