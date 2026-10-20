use ruda_tensor::{ExecutionError, ProfileOptions, ProfileToken, TensorData};
use ruda_communication::{
    Protocol,
    data_service::{TensorDataService, TensorTransferId},
};
use ruda_tensor::graph::{BackendIr, GraphBindings, GraphId, GraphIr, OperationIr, TensorId, TensorIr};
use ruda_tensor_router::{Runner, RunnerClient};
use ruda_core::tensor::DType;
use ruda_core::stream_id::StreamId;
use core::marker::PhantomData;
use std::{collections::HashSet, sync::Arc};
use tokio::sync::mpsc::Sender;
use tokio_util::sync::CancellationToken;

use crate::shared::{ConnectionId, LocalTransferId, SessionId, TaskResponse, TaskResponseContent, TensorRemote};
use super::local_comm::LocalCommService;
use crate::telemetry::{GraphNamespace, GraphOp, TelemetryEvent, TelemetryProbe, TransferPhase, TransferScope, serialized_len};

#[derive(Clone)]
pub(crate) struct WorkerProbe { pub session: SessionId, pub stream: StreamId, pub probe: TelemetryProbe }

struct TransferTrace { observer: Option<WorkerProbe>, scope: TransferScope, peer: Option<String>, complete: bool }
impl TransferTrace {
    fn new(observer: &Option<WorkerProbe>, scope: TransferScope, peer: impl FnOnce() -> Option<String>) -> Self {
        let observer = observer.as_ref().filter(|observer| observer.probe.is_active()).cloned();
        let peer = if observer.is_some() { peer() } else { None };
        let trace = Self { observer, scope, peer, complete: false };
        trace.emit(TransferPhase::Started);
        trace
    }
    fn emit(&self, phase: TransferPhase) {
        if let Some(observer) = &self.observer { observer.probe.emit(|| TelemetryEvent::Transfer {
            session: observer.session, peer: self.peer.clone(), scope: self.scope, phase,
        }); }
    }
    fn complete(&mut self) { self.complete = true; self.emit(TransferPhase::Completed); }
}
impl Drop for TransferTrace {
    fn drop(&mut self) { if !self.complete { self.emit(TransferPhase::Failed); } }
}

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
    RegisterAlias(ConnectionId, TensorId, TensorId, Callback<TaskResponse>),
    QueueGraphReplay(GraphId, Arc<ruda_tensor_router::CachedGraph>, GraphBindings),
    QueueAlias(TensorId, TensorId),
    Flush(ConnectionId, Callback<TaskResponse>),
    ProfileStart(ConnectionId, Callback<TaskResponse>),
    ProfileEnd(ConnectionId, ProfileToken, ProfileOptions, Callback<TaskResponse>),
    ProfileAbandon(ProfileToken),
    DeviceInfo(ConnectionId, crate::RemoteDeviceInfo, Callback<TaskResponse>),
    SyncCollective(ConnectionId, Callback<TaskResponse>),
    ExposeTensorLocal { tensor: TensorIr, transfer_id: LocalTransferId },
    RegisterTensorLocal { transfer_id: LocalTransferId, new_id: TensorId },
    ExposeTensorCapability {
        tensor: TensorIr, count: u32,
        capability: ruda_communication::data_service::TransferCapability,
        target: ruda_communication::PeerId,
    },
    RegisterTensorCapability {
        address: ruda_communication::Address,
        capability: ruda_communication::data_service::TransferCapability,
        new_id: TensorId,
    },
}

struct OpenProfiles<B: BackendIr> {
    runner: Runner<B>,
    stream: StreamId,
    tokens: HashSet<ProfileToken>,
}
impl<B: BackendIr> Drop for OpenProfiles<B> {
    fn drop(&mut self) {
        self.stream.executes(|| {
            for token in self.tokens.drain() { self.runner.profile_abandon(token); }
        });
    }
}

struct WorkerLifetime(CancellationToken);
impl Drop for WorkerLifetime {
    fn drop(&mut self) { self.0.cancel(); }
}

impl<B: BackendIr, P> Processor<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    pub async fn start(
        runner: Runner<B>,
        data_service: Arc<TensorDataService<B, P>>,
        local_comm: Option<(SessionId, Arc<LocalCommService<B>>, CancellationToken)>,
        observer: Option<WorkerProbe>,
    ) -> (Sender<ProcessorTask>, tokio::task::JoinHandle<()>) {
        // channel for tasks to execute
        let (task_sender, mut task_rec) = tokio::sync::mpsc::channel(1);
        let stream = StreamId::unique();

        let worker = tokio::task::spawn_blocking(move || ruda_core::future::block_on(async move {
            let closed = local_comm.as_ref().map(|(_, _, closed)| closed.clone()).unwrap_or_default();
            let _lifetime = WorkerLifetime(closed.clone());
            let mut profiles = OpenProfiles { runner: runner.clone(), stream, tokens: HashSet::new() };
            let mut responses = tokio::task::JoinSet::new();
            while let Some(item) = task_rec.recv().await {
                while let Some(result) = responses.try_join_next() {
                    if let Err(error) = result { log::error!("Remote profile response failed: {error}"); }
                }
                match item {
                    ProcessorTask::RegisterOperation(op) => {
                        if let Some(observer) = &observer { observer.probe.emit(|| TelemetryEvent::unfused_op(observer.session, observer.stream, &op)); }
                        stream.executes(|| runner.register_op(*op));
                    }
                    ProcessorTask::RegisterGraph(id, graph, callback) => {
                        let observed = observer.as_ref().filter(|observer| observer.probe.is_active())
                            .map(|_| (graph.operations.iter().map(GraphOp::summarize).collect(), serialized_len(&graph.operations)));
                        let result = stream.executes(|| runner.register_graph(graph));
                        if let (Some(observer), Some((ops, bytes)), Ok(graph)) = (&observer, observed, &result) {
                            observer.probe.emit(|| TelemetryEvent::GraphRegistered { session: observer.session, graph: *graph,
                                namespace: GraphNamespace::Acknowledged, ops, bytes });
                        }
                        let _ = callback.send(TaskResponse { content: TaskResponseContent::RegisterGraph(result), id }).await;
                    }
                    ProcessorTask::ExecuteGraph(id, graph, bindings, callback) => {
                        let bytes = observer.as_ref().filter(|observer| observer.probe.is_active()).map(|_| serialized_len(&bindings));
                        let result = stream.executes(|| runner.execute_graph(graph, bindings));
                        if result.is_ok() && let (Some(observer), Some(bytes)) = (&observer, bytes) {
                            observer.probe.emit(|| TelemetryEvent::graph_executed(observer.session, graph, GraphNamespace::Acknowledged, observer.stream, bytes));
                        }
                        let _ = callback.send(TaskResponse { content: TaskResponseContent::ExecuteGraph(result), id }).await;
                    }
                    ProcessorTask::RemoveGraph(id, graph, callback) => {
                        let result = stream.executes(|| runner.remove_graph(graph));
                        let _ = callback.send(TaskResponse { content: TaskResponseContent::RemoveGraph(result), id }).await;
                    }
                    ProcessorTask::QueueGraphReplay(id, graph, bindings) => {
                        let bytes = observer.as_ref().filter(|observer| observer.probe.is_active()).map(|_| serialized_len(&bindings));
                        if let Err(error) = stream.executes(|| runner.execute_cached_graph(&graph, bindings)) {
                            log::error!("Remote queued graph replay failed: {error:?}");
                            break;
                        }
                        if let (Some(observer), Some(bytes)) = (&observer, bytes) {
                            observer.probe.emit(|| TelemetryEvent::graph_executed(observer.session, id,
                                crate::telemetry::GraphNamespace::Queued, observer.stream, bytes));
                        }
                    }
                    ProcessorTask::QueueAlias(destination, source) => {
                        if let Err(error) = stream.executes(|| runner.register_alias(destination, source)) {
                            log::error!("Remote queued alias failed: {error:?}");
                            break;
                        }
                    }
                    ProcessorTask::Sync(id, callback) => {
                        if let Some(observer) = &observer { observer.probe.emit(|| TelemetryEvent::Sync { session: observer.session, request: id }); }
                        let result = stream.executes(|| runner.sync());
                        // The responder may already be gone when the client disconnected.
                        let _ = callback
                            .send(TaskResponse {
                                content: TaskResponseContent::SyncBackend(result),
                                id,
                            })
                            .await;
                    }
                    ProcessorTask::RegisterTensor(id, data) => {
                        stream.executes(|| runner.register_tensor_data_id(id, data));
                    }
                    ProcessorTask::ExposeTensorLocal { tensor, transfer_id } => {
                        let mut trace = TransferTrace::new(&observer, TransferScope::Local, || None);
                        let Some((session, service, _)) = &local_comm else {
                            log::error!("Local tensor exposure requires a hosted session");
                            break;
                        };
                        let tensor = stream.executes(|| runner.get_tensor(&tensor));
                        if let Err(error) = service.expose(*session, transfer_id, tensor) {
                            log::error!("Native tensor exposure failed: {error}");
                            break;
                        }
                        trace.complete();
                    }
                    ProcessorTask::RegisterTensorLocal { transfer_id, new_id } => {
                        let mut trace = TransferTrace::new(&observer, TransferScope::Local, || None);
                        let Some((session, service, _)) = &local_comm else {
                            log::error!("Local tensor registration requires a hosted session");
                            break;
                        };
                        match service.take(*session, transfer_id).await {
                            Ok(tensor) => stream.executes(|| runner.register_tensor_to_device(new_id, tensor)),
                            Err(error) => {
                                log::error!("Native tensor registration failed: {error}");
                                break;
                            }
                        }
                        trace.complete();
                    }
                    ProcessorTask::RegisterTensorRemote(remote_tensor, new_id) => {
                        let mut trace = TransferTrace::new(&observer, TransferScope::Remote, || Some(remote_tensor.address.to_string()));
                        log::info!(
                            "Registering remote tensor...(id: {:?})",
                            remote_tensor.transfer_id
                        );
                        let Some(data) = data_service
                            .download_tensor_cancelable(remote_tensor.address, remote_tensor.transfer_id, &closed).await else {
                                log::error!("Remote tensor download failed or the session closed");
                                break;
                            };
                        stream.executes(|| runner.register_tensor_data_id(new_id, data));
                        trace.complete();
                    }
                    ProcessorTask::ExposeTensorRemote {
                        tensor,
                        transfer_id,
                        count,
                    } => {
                        log::info!("Exposing tensor: (id: {transfer_id:?})");
                        let mut trace = TransferTrace::new(&observer, TransferScope::Remote, || None);
                        let read = stream.executes(|| runner.read_tensor_async(tensor));
                        let data = tokio::select! {
                            _ = closed.cancelled() => break,
                            data = read => data,
                        };
                        match data {
                            Ok(data) => data_service.expose_data(data, count, transfer_id).await,
                            Err(error) => { log::error!("Remote tensor readback failed: {error}"); break; }
                        }
                        trace.complete();
                    }
                    ProcessorTask::ExposeTensorCapability { tensor, count, capability, target } => {
                        let mut trace = TransferTrace::new(&observer, TransferScope::Remote, || Some(format!("{target:?}")));
                        let read = stream.executes(|| runner.read_tensor_async(tensor));
                        let data = tokio::select! { _ = closed.cancelled() => break, data = read => data };
                        let result = match data {
                            Ok(data) => data_service.expose_capability(data, count, capability, target, closed.clone()).await,
                            Err(error) => Err(error.to_string()),
                        };
                        if let Err(error) = result { log::error!("Authenticated tensor exposure failed: {error}"); break; }
                        trace.complete();
                    }
                    ProcessorTask::RegisterTensorCapability { address, capability, new_id } => {
                        let mut trace = TransferTrace::new(&observer, TransferScope::Remote, || Some(address.to_string()));
                        let Some(data) = data_service.download_capability(address, capability, &closed).await else {
                            log::error!("Authenticated tensor download failed or the session closed");
                            break;
                        };
                        stream.executes(|| runner.register_tensor_data_id(new_id, data));
                        trace.complete();
                    }
                    ProcessorTask::ReadTensor(id, tensor, callback) => {
                        if let Some(observer) = &observer { observer.probe.emit(|| TelemetryEvent::Read { session: observer.session, request: id }); }
                        let read = stream.executes(|| runner.read_tensor_async(tensor));
                        let tensor = tokio::select! {
                            _ = closed.cancelled() => break,
                            tensor = read => tensor,
                        };
                        let _ = callback
                            .send(TaskResponse {
                                content: TaskResponseContent::ReadTensor(tensor),
                                id,
                            })
                            .await;
                    }
                    ProcessorTask::Close => {
                        break;
                    }
                    ProcessorTask::Seed(seed) => stream.executes(|| runner.seed(seed)),
                    ProcessorTask::DTypeUsage(id, dtype, callback) => {
                        let result = stream.executes(|| runner.dtype_usage(dtype));
                        let _ = callback
                            .send(TaskResponse {
                                content: TaskResponseContent::DTypeUsage(result),
                                id,
                            })
                            .await;
                    }
                    ProcessorTask::RegisterAlias(id, destination, source, callback) => {
                        let result = stream.executes(|| runner.register_alias(destination, source));
                        let _ = callback.send(TaskResponse { content: TaskResponseContent::RegisterAlias(result), id }).await;
                    }
                    ProcessorTask::Flush(id, callback) => {
                        let result = stream.executes(|| runner.flush());
                        let _ = callback.send(TaskResponse { content: TaskResponseContent::FlushBackend(result), id }).await;
                    }
                    ProcessorTask::DeviceInfo(id, info, callback) => {
                        let _ = callback.send(TaskResponse { content: TaskResponseContent::DeviceInfo(info), id }).await;
                    }
                    ProcessorTask::SyncCollective(id, callback) => {
                        let result = if runner.has_native_collectives() {
                            #[cfg(feature = "distributed")]
                            { stream.executes(|| runner.sync_native_collective()); Ok(()) }
                            #[cfg(not(feature = "distributed"))]
                            { Err(ExecutionError::WithContext { reason: "Remote collective support was not compiled".into() }) }
                        } else {
                            Err(ExecutionError::WithContext { reason: "Remote runner has no native collective capability".into() })
                        };
                        let _ = callback.send(TaskResponse { content: TaskResponseContent::SyncCollective(result), id }).await;
                    }
                    ProcessorTask::ProfileStart(id, callback) => {
                        let result = stream.executes(|| runner.profile_start());
                        if let Ok(Some(token)) = &result { profiles.tokens.insert(*token); }
                        let _ = callback.send(TaskResponse { content: TaskResponseContent::ProfileStart(result), id }).await;
                    }
                    ProcessorTask::ProfileEnd(id, token, options, callback) => {
                        let result = if profiles.tokens.remove(&token) {
                            stream.executes(|| runner.profile_end(token, options))
                        } else {
                            Err(ExecutionError::WithContext { reason: "Unknown remote stream profiling window".into() })
                        };
                        responses.spawn_blocking(move || {
                            let result = result.map(|duration| ruda_core::future::block_on(duration.resolve()).map(|ticks| ticks.duration()));
                            let _ = callback.blocking_send(TaskResponse { content: TaskResponseContent::ProfileEnd(result), id });
                        });
                    }
                    ProcessorTask::ProfileAbandon(token) => {
                        if profiles.tokens.remove(&token) { stream.executes(|| runner.profile_abandon(token)); }
                    }
                }
            }
            drop(task_rec);
            closed.cancel();
            drop(profiles);
            let device = runner.device();
            if let Err(error) = stream.executes(|| runner.sync()) {
                log::error!("Failed to sync runner while closing stream: {error}");
            }
            while let Some(result) = responses.join_next().await {
                if let Err(error) = result { log::error!("Remote profile response failed: {error}"); }
            }
            stream.executes(|| core::mem::drop(runner));
            if let Err(error) = stream.executes(|| B::sync(&device)) {
                log::error!("Failed to sync device while closing stream: {error}");
            }
        }));

        (task_sender, worker)
    }
}
