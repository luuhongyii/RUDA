use super::{ActiveChannel, RemoteChannel, RemoteClient};
use crate::shared::{ComputeTask, TaskResponseContent, TensorRemote};
use ruda_tensor::{DeviceId, DeviceOps, ExecutionError, ProfileDuration, ProfileOptions, ProfileTicks, ProfileToken, TensorData};
use ruda_communication::{Address, ProtocolClient, transfer::TensorTransferId};
use ruda_tensor::graph::{GraphBindings, GraphId, GraphIr, TensorIr};
use ruda_tensor_router::{MultiBackendBridge, QueuedGraphId, RouterTensor, RunnerClient, get_client};
use ruda_core::{backtrace::BackTrace, future::DynFut, profile::Instant, stream_id::StreamId};
use std::sync::OnceLock;
use std::{collections::HashMap, marker::PhantomData, str::FromStr, sync::Mutex};

type Endpoint = (Address, u32, crate::Credential);

#[derive(PartialEq, Eq, Hash)]
struct EndpointKey {
    peer: ruda_communication::PeerId,
    application_peer: Option<ruda_communication::PeerId>,
    index: u32,
    credential: crate::Credential,
}

#[derive(Default)]
struct EndpointRegistry {
    by_endpoint: HashMap<EndpointKey, u32>,
    by_id: Vec<Endpoint>,
}

static ADDRESS_REGISTRY: OnceLock<Mutex<EndpointRegistry>> = OnceLock::new();

fn get_address_registry() -> &'static Mutex<EndpointRegistry> {
    ADDRESS_REGISTRY.get_or_init(|| Mutex::new(EndpointRegistry::default()))
}

/// Map a string network address to a (local runtime) global unique u32.
///
/// Globally stable over the lifetime of the process, shared between threads,
/// If the address has never been seen, a new id will be created.
/// If the address has been seen, the previous id will be returned.
pub fn address_to_id<S: AsRef<str>>(address: S) -> u32 {
    endpoint_to_id(&address.as_ref().parse().expect("Invalid remote address"), 0, &crate::Credential::default())
}

fn endpoint_to_id(address: &Address, index: u32, credential: &crate::Credential) -> u32 {
    let key = EndpointKey { peer: address.peer_id(), application_peer: address.application_peer_id(),
        index, credential: credential.clone() };
    let endpoint = (address.clone(), index, credential.clone());
    let mut registry = get_address_registry().lock().unwrap();
    if let Some(id) = registry.by_endpoint.get(&key).copied() {
        registry.by_id[id as usize] = endpoint;
        return id;
    }
    let next_id = registry.by_id.len();
    if next_id > u16::MAX as usize {
        drop(registry);
        panic!("Remote device registry exhausted its DeviceId namespace");
    }
    let id = next_id as u32;
    registry.by_id.push(endpoint);
    registry.by_endpoint.insert(key, id);
    id
}

/// Look up an address by id.
///
/// Returns the same address given ids by [`address_to_id`].
pub fn id_to_address(id: u32) -> Option<String> {
    id_to_endpoint(id).map(|(address, _, _)| address.to_string())
}

fn id_to_endpoint(id: u32) -> Option<Endpoint> {
    get_address_registry().lock().unwrap().by_id.get(id as usize).cloned()
}

// It is very important to block on any request made with the sender, since ordering is crucial
// when registering operation or creating tensors.
//
// The overhead is minimal, since we only wait for the task to be sent to the async
// channel, but not sent to the server and even less processed by the server.
impl RunnerClient for RemoteClient {
    type Device = RemoteDevice;

    fn context_id(&self) -> Option<usize> { Some(std::sync::Arc::as_ptr(&self.sender) as usize) }

    fn register_op(&self, op: ruda_tensor::graph::OperationIr) {
        self.sender
            .send(ComputeTask::RegisterOperation(Box::new(op)));
    }

    fn register_alias(&self, destination: ruda_tensor::graph::TensorId, source: ruda_tensor::graph::TensorId)
        -> Result<(), ExecutionError> {
        match self.graph_request(ComputeTask::RegisterAlias { destination, source })? {
            TaskResponseContent::RegisterAlias(result) => result,
            _ => Err(graph_rpc_error("invalid tensor alias response")),
        }
    }

    fn queue_alias(&self, destination: ruda_tensor::graph::TensorId, source: ruda_tensor::graph::TensorId)
        -> Result<(), ExecutionError> {
        self.queue_compute(ComputeTask::QueueAlias { destination, source })
    }

    fn queue_graph(&self, graph: GraphIr, bindings: GraphBindings) -> Result<QueuedGraphId, ExecutionError> {
        let id = self.sender.new_graph_id().ok_or_else(|| graph_rpc_error("remote queued graph ID space exhausted"))?;
        self.queue_compute(ComputeTask::QueueGraph { id, graph, bindings })?;
        Ok(QueuedGraphId(id))
    }

    fn queue_graph_replay(&self, id: QueuedGraphId, bindings: GraphBindings) -> Result<(), ExecutionError> {
        self.queue_compute(ComputeTask::QueueGraphReplay { id: id.0, bindings })
    }

    fn queue_graph_release(&self, id: QueuedGraphId) -> Result<(), ExecutionError> {
        self.queue_compute(ComputeTask::QueueGraphRelease(id.0))
    }

    fn flush(&self) -> Result<(), ExecutionError> {
        #[cfg(target_family = "wasm")]
        { return self.sender.flush_queue().map_err(|error| graph_rpc_error(format!("Cannot submit remote queue: {error:?}"))); }
        #[cfg(not(target_family = "wasm"))]
        match self.graph_request(ComputeTask::FlushBackend)? {
            TaskResponseContent::FlushBackend(result) => result,
            _ => Err(graph_rpc_error("invalid backend submission response")),
        }
    }

    fn profile_start(&self) -> Result<Option<ProfileToken>, ExecutionError> {
        #[cfg(target_family = "wasm")]
        { return Err(graph_rpc_error("a browser thread cannot wait for a remote profiling window")); }
        #[cfg(not(target_family = "wasm"))]
        {
            let stream = StreamId::current();
            let id = {
                let mut profiles = self.profiles.lock().unwrap();
                let id = profiles.counter;
                profiles.counter = id.checked_add(1).ok_or_else(|| graph_rpc_error("remote profiling token space exhausted"))?;
                id
            };
            let future = self.sender.send_async_on(ComputeTask::ProfileStart, stream);
            let response = self.runtime.block_on(future)
                .map_err(|error| graph_rpc_error(format!("profiling start request failed: {error:?}")))?;
            match response {
                TaskResponseContent::ProfileStart(result) => {
                    let Some(inner) = result? else { return Ok(None); };
                    self.profiles.lock().unwrap().open.insert(id, (stream, inner));
                    Ok(Some(ProfileToken { id, opened_on: stream.value }))
                }
                _ => Err(graph_rpc_error("invalid profiling start response")),
            }
        }
    }

    fn profile_end(&self, token: ProfileToken, options: ProfileOptions) -> Result<ProfileDuration, ExecutionError> {
        let Some((stream, inner)) = self.take_profile(token) else {
            log::error!("Unknown remote profiling window: {token:?}");
            return Ok(ProfileDuration::new_device_time_maybe(async { None }));
        };
        let future = self.sender.send_async_on(ComputeTask::ProfileEnd(inner, options), stream);
        let client = self.clone();
        Ok(ProfileDuration::new_device_time_maybe(async move {
            let _client = client;
            match future.await {
                Ok(TaskResponseContent::ProfileEnd(Ok(Some(duration)))) => {
                    let start = Instant::now();
                    Some(ProfileTicks::from_start_end(start, start + duration))
                }
                Ok(TaskResponseContent::ProfileEnd(Ok(None))) => None,
                other => {
                    log::error!("Remote profiling window returned no measurement: {other:?}");
                    None
                }
            }
        }))
    }

    fn profile_abandon(&self, token: ProfileToken) {
        if let Some((stream, inner)) = self.take_profile(token) {
            if let Err(error) = self.sender.send_on(ComputeTask::ProfileAbandon(inner), stream) {
                log::warn!("Could not abandon remote profiling window: {error:?}");
            }
        }
    }

    fn register_graph(&self, graph: GraphIr) -> Result<GraphId, ExecutionError> {
        match self.graph_request(ComputeTask::RegisterGraph(graph))? {
            TaskResponseContent::RegisterGraph(result) => result,
            _ => Err(graph_rpc_error("invalid graph registration response")),
        }
    }

    fn execute_graph(&self, graph: GraphId, bindings: GraphBindings) -> Result<(), ExecutionError> {
        match self.graph_request(ComputeTask::ExecuteGraph(graph, bindings))? {
            TaskResponseContent::ExecuteGraph(result) => result,
            _ => Err(graph_rpc_error("invalid graph execution response")),
        }
    }

    fn remove_graph(&self, graph: GraphId) -> Result<(), ExecutionError> {
        match self.graph_request(ComputeTask::RemoveGraph(graph))? {
            TaskResponseContent::RemoveGraph(result) => result,
            _ => Err(graph_rpc_error("invalid graph release response")),
        }
    }

    fn read_tensor_async(
        &self,
        tensor: ruda_tensor::graph::TensorIr,
    ) -> DynFut<Result<TensorData, ExecutionError>> {
        // Important for ordering to call the creation of the future sync.
        let fut = self.sender.send_async(ComputeTask::ReadTensor(tensor));
        let client = self.clone();

        Box::pin(async move {
            let _client = client;
            match fut.await {
                Ok(response) => match response {
                    TaskResponseContent::ReadTensor(res) => res,
                    _ => panic!("Invalid message type"),
                },
                Err(e) => Err(ExecutionError::Generic {
                    reason: format!("Failed to read tensor: {:?}", e),
                    backtrace: BackTrace::capture(),
                }),
            }
        })
    }

    fn register_tensor_data(&self, data: TensorData) -> RouterTensor<Self> {
        let id = self.sender.new_tensor_id();
        let shape = data.shape.clone();
        let dtype = data.dtype;

        self.sender.send(ComputeTask::RegisterTensor(id, data));

        RouterTensor::new(id, shape, dtype, self.clone())
    }

    fn device(&self) -> Self::Device {
        self.device.clone()
    }

    fn sync(&self) -> Result<(), ExecutionError> {
        #[cfg(target_family = "wasm")]
        { return self.flush(); }
        #[cfg(not(target_family = "wasm"))]
        {
        // Important for ordering to call the creation of the future sync.
        let fut = self.sender.send_async(ComputeTask::SyncBackend);

        match self.runtime.block_on(fut) {
            Ok(response) => match response {
                TaskResponseContent::SyncBackend(res) => res,
                _ => panic!("Invalid message type"),
            },
            Err(e) => Err(ExecutionError::Generic {
                reason: format!("Failed to sync: {:?}", e),
                backtrace: BackTrace::capture(),
            }),
        }
        }
    }

    fn seed(&self, seed: u64) {
        self.sender.send(ComputeTask::Seed(seed));
    }

    fn rng_state(&self) -> Result<Vec<u8>, ExecutionError> {
        match self.graph_request(ComputeTask::RngState)? {
            TaskResponseContent::RngState(result) => result,
            _ => Err(graph_rpc_error("Invalid RNG checkpoint response")),
        }
    }

    fn restore_rng_state(&self, state: &[u8]) -> Result<(), ExecutionError> {
        match self.graph_request(ComputeTask::RestoreRngState(state.to_vec()))? {
            TaskResponseContent::RestoreRngState(result) => result,
            _ => Err(graph_rpc_error("Invalid RNG restore response")),
        }
    }

    fn create_empty_handle(&self) -> ruda_tensor::graph::TensorId {
        self.sender.new_tensor_id()
    }

    fn dtype_usage(&self, dtype: ruda_core::tensor::DType) -> ruda_tensor::DTypeUsageSet {
        let fut = self.sender.send_async(ComputeTask::DTypeUsage(dtype));

        match self.runtime.block_on(fut) {
            Ok(response) => match response {
                TaskResponseContent::DTypeUsage(res) => res,
                other => panic!("Invalid message type {other:?}"),
            },
            Err(e) => panic!("Failed to check dtype support: {:?}", e),
        }
    }
}

fn graph_rpc_error(reason: impl Into<String>) -> ExecutionError {
    ExecutionError::Generic { reason: reason.into(), backtrace: BackTrace::capture() }
}
impl RemoteClient {
    fn request_async(&self, task: ComputeTask)
        -> impl core::future::Future<Output = Result<TaskResponseContent, ExecutionError>> + Send + use<> {
        let response = self.sender.send_async(task);
        let client = self.clone();
        async move {
            let _client = client;
            response.await.map_err(|error| graph_rpc_error(format!("Remote request failed: {error:?}")))
        }
    }

    /// Wait for device completion without blocking the browser event loop.
    pub fn sync_async(&self) -> impl core::future::Future<Output = Result<(), ExecutionError>> + Send + use<> {
        let response = self.request_async(ComputeTask::SyncBackend);
        async move { match response.await? {
            TaskResponseContent::SyncBackend(result) => result,
            _ => Err(graph_rpc_error("Invalid synchronization response")),
        } }
    }

    pub fn rng_state_async(&self)
        -> impl core::future::Future<Output = Result<Vec<u8>, ExecutionError>> + Send + use<> {
        let response = self.request_async(ComputeTask::RngState);
        async move { match response.await? {
            TaskResponseContent::RngState(result) => result,
            _ => Err(graph_rpc_error("Invalid RNG checkpoint response")),
        } }
    }

    pub fn restore_rng_state_async(&self, state: &[u8])
        -> impl core::future::Future<Output = Result<(), ExecutionError>> + Send + use<> {
        let response = self.request_async(ComputeTask::RestoreRngState(state.to_vec()));
        async move { match response.await? {
            TaskResponseContent::RestoreRngState(result) => result,
            _ => Err(graph_rpc_error("Invalid RNG restore response")),
        } }
    }

    pub fn flush_async(&self) -> impl core::future::Future<Output = Result<(), ExecutionError>> + Send + use<> {
        let response = self.request_async(ComputeTask::FlushBackend);
        async move { match response.await? {
            TaskResponseContent::FlushBackend(result) => result,
            _ => Err(graph_rpc_error("Invalid submission response")),
        } }
    }

    pub fn dtype_usage_async(&self, dtype: ruda_tensor::DType)
        -> impl core::future::Future<Output = Result<ruda_tensor::DTypeUsageSet, ExecutionError>> + Send + use<> {
        let response = self.request_async(ComputeTask::DTypeUsage(dtype));
        async move { match response.await? {
            TaskResponseContent::DTypeUsage(result) => Ok(result),
            _ => Err(graph_rpc_error("Invalid dtype-usage response")),
        } }
    }

    pub fn register_graph_async(&self, graph: GraphIr)
        -> impl core::future::Future<Output = Result<GraphId, ExecutionError>> + Send + use<> {
        let response = self.request_async(ComputeTask::RegisterGraph(graph));
        async move { match response.await? {
            TaskResponseContent::RegisterGraph(result) => result,
            _ => Err(graph_rpc_error("Invalid graph-registration response")),
        } }
    }

    pub fn execute_graph_async(&self, graph: GraphId, bindings: GraphBindings)
        -> impl core::future::Future<Output = Result<(), ExecutionError>> + Send + use<> {
        let response = self.request_async(ComputeTask::ExecuteGraph(graph, bindings));
        async move { match response.await? {
            TaskResponseContent::ExecuteGraph(result) => result,
            _ => Err(graph_rpc_error("Invalid graph-execution response")),
        } }
    }

    pub fn remove_graph_async(&self, graph: GraphId)
        -> impl core::future::Future<Output = Result<(), ExecutionError>> + Send + use<> {
        let response = self.request_async(ComputeTask::RemoveGraph(graph));
        async move { match response.await? {
            TaskResponseContent::RemoveGraph(result) => result,
            _ => Err(graph_rpc_error("Invalid graph-removal response")),
        } }
    }

    pub fn register_alias_async(&self, destination: ruda_tensor::graph::TensorId, source: ruda_tensor::graph::TensorId)
        -> impl core::future::Future<Output = Result<(), ExecutionError>> + Send + use<> {
        let response = self.request_async(ComputeTask::RegisterAlias { destination, source });
        async move { match response.await? {
            TaskResponseContent::RegisterAlias(result) => result,
            _ => Err(graph_rpc_error("Invalid tensor-alias response")),
        } }
    }

    /// Await the versioned handshake, including both session connections.
    pub async fn connected(&self) -> Result<crate::RemoteDeviceInfo, super::ConnectError> {
        let info = self.connection.wait().await?;
        let _ = self.info.set(info.clone());
        Ok(info)
    }

    /// Read the selected server device's metadata, retaining it for subsequent queries.
    pub async fn device_info_async(&self) -> Result<crate::RemoteDeviceInfo, ExecutionError> {
        if let Some(info) = self.info.get() { return Ok(info.clone()); }
        let info = self.connected().await.map_err(|error| graph_rpc_error(error.to_string()))?;
        let _ = self.info.set(info.clone());
        Ok(info)
    }

    pub fn device_info(&self) -> Result<crate::RemoteDeviceInfo, ExecutionError> {
        if let Some(info) = self.info.get() { return Ok(info.clone()); }
        self.runtime.block_on(self.device_info_async())
    }

    #[cfg(feature = "distributed")]
    pub(crate) fn sync_native_collective(&self) -> Result<(), ExecutionError> {
        match self.graph_request(ComputeTask::SyncCollective)? {
            TaskResponseContent::SyncCollective(result) => result,
            _ => Err(graph_rpc_error("invalid collective synchronization response")),
        }
    }

    fn take_profile(&self, token: ProfileToken) -> Option<(StreamId, ProfileToken)> {
        let mut profiles = self.profiles.lock().unwrap();
        if profiles.open.get(&token.id).is_some_and(|(stream, _)| stream.value == token.opened_on) {
            profiles.open.remove(&token.id)
        } else { None }
    }

    fn graph_request(&self, task: ComputeTask) -> Result<TaskResponseContent, ExecutionError> {
        let future = self.sender.send_async(task);
        self.runtime.block_on(future).map_err(|error| graph_rpc_error(format!("remote request failed: {error:?}")))
    }

    fn queue_compute(&self, task: ComputeTask) -> Result<(), ExecutionError> {
        self.sender.send_on(task, StreamId::current())
            .map_err(|error| graph_rpc_error(format!("Cannot queue remote computation: {error:?}")))
    }
}

#[derive(Clone, Debug)]
/// The device contains the connection information of the server.
pub struct RemoteDevice {
    pub(crate) address: Address,
    pub(crate) device_index: u32,
    pub(crate) credential: crate::Credential,
    /// The id of the device in the local registry, see [`address_to_id`].
    pub(crate) id: u32,
}

impl RemoteDevice {
    /// Create a device from an url.
    pub fn new(address: &str) -> Self {
        Self::with_index(address, 0)
    }

    /// Select a position in the server's configured device list, connecting on first use.
    pub fn with_index(address: &str, device_index: usize) -> Self {
        Self::with_credential(address, device_index, crate::Credential::default())
    }

    pub(crate) fn with_credential(address: &str, device_index: usize, credential: crate::Credential) -> Self {
        Self::with_address(Address::from_str(address).expect("Invalid remote address"), device_index, credential)
    }

    pub(crate) fn with_address(address: Address, device_index: usize, credential: crate::Credential) -> Self {
        let device_index = u32::try_from(device_index).expect("Remote device index exceeds u32");
        let id = endpoint_to_id(&address, device_index, &credential);
        Self {
            address,
            device_index,
            credential,
            id,
        }
    }

    pub fn device_index(&self) -> usize { self.device_index as usize }

    pub(crate) fn refreshed(&self) -> Self {
        let (address, device_index, credential) = id_to_endpoint(self.id)
            .expect("Remote device is not registered");
        Self { address, device_index, credential, id: self.id }
    }
}

impl PartialEq for RemoteDevice {
    fn eq(&self, other: &Self) -> bool { self.id == other.id }
}
impl Eq for RemoteDevice {}

impl Default for RemoteDevice {
    fn default() -> Self {
        let address = match std::env::var("RUDA_REMOTE_ADDRESS") {
            Ok(address) => address,
            Err(_) => String::from("ws://127.0.0.1:3000"),
        };

        Self::new(&address)
    }
}

impl ruda_core::device::Device for RemoteDevice {
    fn from_id(device_id: DeviceId) -> Self {
        if device_id.type_id != 0 {
            panic!("Invalid device id: {device_id} (expected type 0)");
        }
        let (address, index, credential) = id_to_endpoint(device_id.index_id as u32)
            .unwrap_or_else(|| panic!("Invalid device id: {device_id}"));
        Self { address, device_index: index, credential, id: device_id.index_id as u32 }
    }

    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: 0,
            index_id: self.id as u16,
        }
    }
}

impl DeviceOps for RemoteDevice {
    fn defaults(&self) -> Option<ruda_tensor::DeviceSettings> {
        let info = crate::remote_client(self).device_info()
            .unwrap_or_else(|error| panic!("Cannot obtain remote device defaults: {error:?}"));
        Some(ruda_tensor::DeviceSettings::new(info.float_dtype, info.int_dtype, info.bool_dtype))
    }
}

pub struct RemoteBridge<C: ProtocolClient> {
    _p: PhantomData<C>,
}

pub struct RemoteTensorHandle<C: ProtocolClient> {
    pub(crate) client: RemoteClient,
    pub(crate) tensor: TensorIr,
    pub(crate) _p: PhantomData<C>,
}

fn get_next_transfer_id() -> TensorTransferId {
    ruda_core::id::IdGenerator::generate().into()
}

impl<C: ProtocolClient> RemoteTensorHandle<C> {
    /// Transfer within a server using native primitives, or between servers using the data service.
    pub(crate) fn change_backend(self, target_device: &RemoteDevice) -> Self {
        if self.client.device.address.same_peer(&target_device.address) {
            self.change_backend_local(target_device)
        } else {
            self.change_backend_remote(target_device)
        }
    }

    fn change_backend_local(mut self, target_device: &RemoteDevice) -> Self {
        let target_client = get_client::<ActiveChannel<C>>(target_device);
        // The first metadata response acknowledges session registration. Both sessions must
        // exist before either side enters the rendezvous; subsequent calls use cached metadata.
        self.client.device_info().expect("Cannot initialize local transfer source");
        target_client.device_info().expect("Cannot initialize local transfer target");
        let transfer_id = self.client.sender.new_local_transfer(&target_client.sender);
        let new_id = target_client.sender.new_tensor_id();
        self.client.sender.send(ComputeTask::ExposeTensorLocal {
            tensor: self.tensor.clone(), transfer_id,
        });
        target_client.sender.send(ComputeTask::RegisterTensorLocal { transfer_id, new_id });
        self.tensor.id = new_id;
        self.client = target_client;
        self
    }

    /// Changes the backend of the tensor through the server-to-server data service.
    /// We ask the original server to expose the tensor, then ask the target server to fetch
    /// the tensor. The target server will open a new network connection to the original server
    /// to download the data.
    /// This way the client never sees the tensor's data, and we avoid a bottleneck.
    fn change_backend_remote(mut self, target_device: &RemoteDevice) -> Self {
        use ruda_communication::{PeerId, transfer::TransferCapability};
        match (self.client.device.address.peer_id(), target_device.address.peer_id()) {
            (PeerId::Iroh(_), target @ PeerId::Iroh(_)) => {
                let target_client = get_client::<ActiveChannel<C>>(target_device);
                let new_id = target_client.sender.new_tensor_id();
                let capability = TransferCapability::random();
                self.client.sender.send(ComputeTask::ExposeTensorCapability { tensor: self.tensor.clone(), count: 1, capability, target });
                target_client.sender.send(ComputeTask::RegisterTensorCapability {
                    address: self.client.device.address.clone(), capability, new_id,
                });
                self.tensor.id = new_id;
                self.client = target_client;
                return self;
            }
            (PeerId::WebSocket(_), PeerId::WebSocket(_)) => {},
            _ => panic!("Remote tensor transfer requires matching server transports"),
        }
        let transfer_id = get_next_transfer_id();
        self.client.sender.send(ComputeTask::ExposeTensorRemote {
            tensor: self.tensor.clone(),
            count: 1,
            transfer_id,
        });

        let target_client = get_client::<ActiveChannel<C>>(target_device);

        let new_id = target_client.sender.new_tensor_id();

        let remote_tensor = TensorRemote {
            transfer_id,
            address: self.client.device.address.clone(),
        };
        target_client
            .sender
            .send(ComputeTask::RegisterTensorRemote(remote_tensor, new_id));

        self.tensor.id = new_id;
        self.client = target_client;

        self
    }
}

impl<C: ProtocolClient> MultiBackendBridge for RemoteBridge<C> {
    type TensorHandle = RemoteTensorHandle<C>;
    type Device = RemoteDevice;

    fn change_backend_quantized(
        tensor: Self::TensorHandle,
        _shape: ruda_tensor::Shape,
        target_device: &Self::Device,
    ) -> Self::TensorHandle {
        tensor.change_backend(target_device)
    }

    fn change_backend_float(
        tensor: Self::TensorHandle,
        _shape: ruda_tensor::Shape,
        target_device: &Self::Device,
    ) -> Self::TensorHandle {
        tensor.change_backend(target_device)
    }

    fn change_backend_int(
        tensor: Self::TensorHandle,
        _shape: ruda_tensor::Shape,
        target_device: &Self::Device,
    ) -> Self::TensorHandle {
        tensor.change_backend(target_device)
    }

    fn change_backend_bool(
        tensor: Self::TensorHandle,
        _shape: ruda_tensor::Shape,
        target_device: &Self::Device,
    ) -> Self::TensorHandle {
        tensor.change_backend(target_device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_address_to_id() {
        let address1 = "ws://127.0.0.1:3000";
        let address2 = "ws://127.0.0.1:3001";

        let id1 = address_to_id(address1);
        let id2 = address_to_id(address2);

        assert_ne!(id1, id2);

        assert_eq!(address_to_id(address1), id1);
        assert_eq!(id_to_address(id1), Some(address1.to_string()));

        assert_eq!(address_to_id(address2), id2);
        assert_eq!(id_to_address(id2), Some(address2.to_string()));

        let unused_id = u32::MAX;

        assert_eq!(id_to_address(unused_id), None);
    }
}
