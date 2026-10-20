use ruda_tensor::{DTypeUsageSet, ExecutionError, ProfileOptions, ProfileToken, TensorData};
use ruda_communication::{Address, transfer::TensorTransferId};
use ruda_tensor::graph::{GraphBindings, GraphId, GraphIr, OperationIr, TensorId, TensorIr};
use ruda_core::{
    tensor::DType,
    id::{IdGenerator, StreamId},
};
use serde::{Deserialize, Serialize};
use std::fmt::Display;

/// Version of the ruDa remote application protocol, independent of the transport.
pub const PROTOCOL_VERSION: u16 = 2;

#[derive(Clone, Serialize, Deserialize)]
pub struct SessionInit {
    pub version: u16,
    pub session_id: SessionId,
    pub device_index: u32,
    #[serde(with = "serde_bytes")]
    pub authorization: Vec<u8>,
}
impl core::fmt::Debug for SessionInit {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SessionInit").field("version", &self.version)
            .field("session_id", &self.session_id).field("device_index", &self.device_index)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SessionRefusal {
    Unauthorized,
    NoSuchDevice { device_count: u32 },
    IncompatibleProtocol { server_version: u16 },
    InvalidSession,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum SessionReply {
    Accepted { version: u16, device: RemoteDeviceInfo },
    Refused(SessionRefusal),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SessionClosed { pub session: SessionId }

#[allow(missing_docs)]
#[derive(new, Serialize, Deserialize, Debug, Hash, PartialEq, Eq, Clone, Copy, PartialOrd, Ord)]
pub struct ConnectionId {
    pub position: u64,
    pub stream_id: StreamId,
}

/// Unique identifier that can represent a session.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct SessionId {
    id: u64,
}

impl Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "SessionId({})", self.id)
    }
}

impl SessionId {
    /// Create a new [session id](SessionId).
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            id: IdGenerator::generate(),
        }
    }
}

/// A same-server transfer, scoped to both endpoint sessions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct LocalTransferId {
    pub source: SessionId,
    pub target: SessionId,
    pub sequence: u64,
}

#[allow(missing_docs)]
#[derive(Serialize, Deserialize, Debug)]
pub enum Task {
    Compute(ComputeTask, ConnectionId),
    Init(SessionId),
    Close(SessionId),
    InitDevice(SessionId, u32),
    InitSession(SessionInit),
    Batch(Vec<Task>),
}

/// Information about a selected device and the server's ordered device catalog.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteDeviceInfo {
    pub device_index: u32,
    pub device_count: u32,
    pub name: String,
    pub float_dtype: ruda_tensor::FloatDType,
    pub int_dtype: ruda_tensor::IntDType,
    pub bool_dtype: ruda_tensor::BoolDType,
    pub native_collectives: bool,
}

#[allow(missing_docs)]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TensorRemote {
    pub transfer_id: TensorTransferId,
    pub address: Address,
}

#[allow(missing_docs)]
#[derive(Serialize, Deserialize, Debug)]
pub enum ComputeTask {
    Seed(u64),
    RegisterOperation(Box<OperationIr>),
    RegisterTensor(TensorId, TensorData),
    RegisterTensorRemote(TensorRemote, TensorId),
    ExposeTensorRemote {
        tensor: TensorIr,
        count: u32,
        transfer_id: TensorTransferId,
    },
    ReadTensor(TensorIr),
    SyncBackend,
    DTypeUsage(DType),
    RegisterGraph(GraphIr),
    ExecuteGraph(GraphId, GraphBindings),
    RemoveGraph(GraphId),
    RegisterAlias { destination: TensorId, source: TensorId },
    FlushBackend,
    ProfileStart,
    ProfileEnd(ProfileToken, ProfileOptions),
    ProfileAbandon(ProfileToken),
    DeviceInfo,
    SyncCollective,
    ExposeTensorLocal { tensor: TensorIr, transfer_id: LocalTransferId },
    RegisterTensorLocal { transfer_id: LocalTransferId, new_id: TensorId },
    ExposeTensorCapability {
        tensor: TensorIr,
        count: u32,
        capability: ruda_communication::transfer::TransferCapability,
        target: ruda_communication::PeerId,
    },
    RegisterTensorCapability {
        address: Address,
        capability: ruda_communication::transfer::TransferCapability,
        new_id: TensorId,
    },
    QueueGraph { id: GraphId, graph: GraphIr, bindings: GraphBindings },
    QueueGraphReplay { id: GraphId, bindings: GraphBindings },
    QueueGraphRelease(GraphId),
    QueueAlias { destination: TensorId, source: TensorId },
}

#[allow(missing_docs)]
#[derive(Serialize, Deserialize, Debug)]
pub struct TaskResponse {
    pub content: TaskResponseContent,
    pub id: ConnectionId,
}

#[allow(missing_docs)]
#[derive(Serialize, Deserialize, Debug)]
pub enum TaskResponseContent {
    ReadTensor(Result<TensorData, ExecutionError>),
    SyncBackend(Result<(), ExecutionError>),
    DTypeUsage(DTypeUsageSet),
    RegisterGraph(Result<GraphId, ExecutionError>),
    ExecuteGraph(Result<(), ExecutionError>),
    RemoveGraph(Result<(), ExecutionError>),
    RegisterAlias(Result<(), ExecutionError>),
    FlushBackend(Result<(), ExecutionError>),
    ProfileStart(Result<Option<ProfileToken>, ExecutionError>),
    ProfileEnd(Result<Option<core::time::Duration>, ExecutionError>),
    DeviceInfo(RemoteDeviceInfo),
    SyncCollective(Result<(), ExecutionError>),
}
