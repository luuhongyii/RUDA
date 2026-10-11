use ruda_tensor::TensorData;
use ruda_communication::{
    Protocol,
    data_service::{TensorDataService, TensorTransferId},
};
use ruda_tensor::graph::{BackendIr, GraphBindings, GraphId, GraphIr, OperationIr, TensorId, TensorIr};
use ruda_tensor_router::{Runner, RunnerClient};
use ruda_core::tensor::DType;
use core::marker::PhantomData;
use std::sync::Arc;
use tokio::sync::mpsc::Sender;

use crate::shared::{ConnectionId, TaskResponse, TaskResponseContent, TensorRemote};

/// The goal of the processor is to asynchronously process compute tasks on it own thread.
pub struct Processor<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    p: PhantomData<B>,
    n: PhantomData<P>,
}

pub type Callback<M> = Sender<M>;

pub enum ProcessorTask {
    RegisterOperation(Box<OperationIr>),
    RegisterTensor(TensorId, TensorData),
    RegisterTensorRemote(TensorRemote, TensorId),
    ExposeTensorRemote {
        tensor: TensorIr,
        transfer_id: TensorTransferId,
        count: u32,
    },
    ReadTensor(ConnectionId, TensorIr, Callback<TaskResponse>),
    Sync(ConnectionId, Callback<TaskResponse>),
    Seed(u64),
    DTypeUsage(ConnectionId, DType, Callback<TaskResponse>),
    Close,
    RegisterGraph(ConnectionId, GraphIr, Callback<TaskResponse>),
    ExecuteGraph(ConnectionId, GraphId, GraphBindings, Callback<TaskResponse>),
    RemoveGraph(ConnectionId, GraphId, Callback<TaskResponse>),
}

impl<B: BackendIr, P> Processor<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    pub async fn start(
        runner: Runner<B>,
        data_service: Arc<TensorDataService<B, P>>,
    ) -> (Sender<ProcessorTask>, tokio::task::JoinHandle<()>) {
        // channel for tasks to execute
        let (task_sender, mut task_rec) = tokio::sync::mpsc::channel(1);

        let worker = tokio::spawn(async move {
            while let Some(item) = task_rec.recv().await {
                match item {
                    ProcessorTask::RegisterOperation(op) => {
                        runner.register_op(*op);
                    }
                    ProcessorTask::RegisterGraph(id, graph, callback) => {
                        let result = runner.register_graph(graph);
                        let _ = callback.send(TaskResponse { content: TaskResponseContent::RegisterGraph(result), id }).await;
                    }
                    ProcessorTask::ExecuteGraph(id, graph, bindings, callback) => {
                        let result = runner.execute_graph(graph, bindings);
                        let _ = callback.send(TaskResponse { content: TaskResponseContent::ExecuteGraph(result), id }).await;
                    }
                    ProcessorTask::RemoveGraph(id, graph, callback) => {
                        let result = runner.remove_graph(graph);
                        let _ = callback.send(TaskResponse { content: TaskResponseContent::RemoveGraph(result), id }).await;
                    }
                    ProcessorTask::Sync(id, callback) => {
                        let result = runner.sync();
                        // The responder may already be gone when the client disconnected.
                        let _ = callback
                            .send(TaskResponse {
                                content: TaskResponseContent::SyncBackend(result),
                                id,
                            })
                            .await;
                    }
                    ProcessorTask::RegisterTensor(id, data) => {
                        runner.register_tensor_data_id(id, data);
                    }
                    ProcessorTask::RegisterTensorRemote(remote_tensor, new_id) => {
                        log::info!(
                            "Registering remote tensor...(id: {:?})",
                            remote_tensor.transfer_id
                        );
                        let data = data_service
                            .download_tensor(remote_tensor.address, remote_tensor.transfer_id)
                            .await
                            .expect("Can't download tensor: error"); // TODO all these panics should be server errors
                        runner.register_tensor_data_id(new_id, data);
                    }
                    ProcessorTask::ExposeTensorRemote {
                        tensor,
                        transfer_id,
                        count,
                    } => {
                        log::info!("Exposing tensor: (id: {transfer_id:?})");
                        let data = runner.read_tensor_async(tensor).await;
                        data_service
                            .expose_data(data.unwrap(), count, transfer_id)
                            .await;
                    }
                    ProcessorTask::ReadTensor(id, tensor, callback) => {
                        let tensor = runner.read_tensor_async(tensor).await;
                        let _ = callback
                            .send(TaskResponse {
                                content: TaskResponseContent::ReadTensor(tensor),
                                id,
                            })
                            .await;
                    }
                    ProcessorTask::Close => {
                        let device = runner.device();
                        if let Err(err) = runner.sync() {
                            log::error!("Failed to sync runner while closing stream: {err:?}");
                        }
                        core::mem::drop(runner);
                        if let Err(err) = B::sync(&device) {
                            log::error!("Failed to sync device while closing stream: {err:?}");
                        }
                        break;
                    }
                    ProcessorTask::Seed(seed) => runner.seed(seed),
                    ProcessorTask::DTypeUsage(id, dtype, callback) => {
                        let result = runner.dtype_usage(dtype);
                        let _ = callback
                            .send(TaskResponse {
                                content: TaskResponseContent::DTypeUsage(result),
                                id,
                            })
                            .await;
                    }
                }
            }
        });

        (task_sender, worker)
    }
}
