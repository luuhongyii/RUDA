use super::Handle;
use crate::runtime::{
    client::ComputeClient,
    compiler::CompilationError,
    config::{RudaRuntimeConfig, RuntimeConfig, compilation::BoundsCheckMode},
    kernel::KernelMetadata,
    logging::ServerLogger,
    memory_management::{ManagedMemoryHandle, MemoryAllocationMode, MemoryUsage},
    backend::Runtime,
    server::Binding,
    storage::{ComputeStorage, ManagedResource},
    tma::{OobFill, TensorMapFormat, TensorMapInterleave, TensorMapPrefetch, TensorMapSwizzle},
};
use alloc::boxed::Box;
#[cfg(feature = "runtime-profile-tracy")]
use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt::Debug;
use ruda_core::{
    backtrace::BackTrace,
    bytes::Bytes,
    device::{self, DeviceId, ServiceId},
    future::DynFut,
    profile::ProfileDuration,
    stream_id::StreamId,
    stub::RwLock,
};
use ruda_core::ir::{DeviceProperties, ElemType, StorageType};
use ruda_core::tensor::{Shape, Strides, metadata::Metadata};
use hashbrown::HashSet;
use thiserror::Error;
pub use ruda_core::poison::DevicePoison;

#[derive(Error, Clone)]
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
/// An error during profiling.
pub enum ProfileError {
    /// An unknown error happened during profiling
    #[error(
        "An unknown error happened during profiling\nCaused by:\n  {reason}\nBacktrace:\n{backtrace}"
    )]
    Unknown {
        /// The caused of the error
        reason: String,
        /// The captured backtrace.
        #[cfg_attr(serializable, serde(skip))]
        backtrace: BackTrace,
    },

    /// No profiling was registered
    #[error("No profiling registered\nBacktrace:\n{backtrace}")]
    NotRegistered {
        /// The captured backtrace.
        #[cfg_attr(serializable, serde(skip))]
        backtrace: BackTrace,
    },

    /// A launch error happened during profiling
    #[error("A launch error happened during profiling\nCaused by:\n  {0}")]
    Launch(#[from] LaunchError),

    /// An execution error happened during profiling
    #[error("An execution error happened during profiling\nCaused by:\n  {0}")]
    Server(#[from] Box<ServerError>),

    /// The window contained no device measurement.
    #[error("Profiling window contained no measurement\nBacktrace:\n{backtrace}")]
    NotMeasured {
        /// The captured backtrace.
        #[cfg_attr(serializable, serde(skip))]
        backtrace: BackTrace,
    },
}

impl core::fmt::Debug for ProfileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self}"))
    }
}

impl ProfileError {
    /// Whether a fault in the measured work poisoned its device.
    pub fn is_device_poisoned(&self) -> bool {
        match self {
            Self::Launch(error) => error.is_device_poisoned(),
            Self::Server(error) => error.is_device_poisoned(),
            _ => false,
        }
    }

    /// Whether the measured work was refused before execution.
    pub fn is_refusal(&self) -> bool {
        match self {
            Self::Launch(error) => error.is_refusal(),
            Self::Server(error) => error.is_refusal(),
            _ => false,
        }
    }
}

/// Contains many different types that are useful for server implementations and compute clients.
pub struct ServerUtilities<Server: ComputeServer> {
    /// Identity stamped on allocations made by this service.
    pub service: ServiceId,
    /// The time when `profile-tracy` is activated.
    #[cfg(feature = "runtime-profile-tracy")]
    pub epoch_time: web_time::Instant,
    /// The GPU client when `profile-tracy` is activated.
    #[cfg(feature = "runtime-profile-tracy")]
    pub gpu_client: tracy_client::GpuContext,
    /// Information shared between all servers.
    pub properties: DeviceProperties,
    /// Stable hash of the device properties
    pub properties_hash: u64,
    /// Information specific to the current server.
    pub info: Server::Info,
    pub runtime_name: Option<&'static str>,
    /// The logger based on global ruda configs.
    pub logger: Arc<ServerLogger>,
    /// How to create the allocation.
    pub layout_policy: Server::MemoryLayoutPolicy,
    /// How to enforce bounds checking on kernels.
    pub check_mode: BoundsCheckMode,
    /// A set containing the ids for which the inter-device communication has already been initialized.
    pub initialized_comms: RwLock<HashSet<CommunicationId>>,
}

/// Defines how the memory layout is determined.
pub trait MemoryLayoutPolicy: Send + Sync + 'static {
    /// Applies the memory layout policy to a list of descriptors.
    ///
    /// Returns a vector of `MemoryLayout`, one per descriptor, with layouts that share a
    /// single `Binding`.
    fn apply(
        &self,
        service: ServiceId,
        stream_id: StreamId,
        descriptors: &[MemoryLayoutDescriptor],
    ) -> (Handle, Vec<MemoryLayout>);
}

impl<Server: core::fmt::Debug> core::fmt::Debug for ServerUtilities<Server>
where
    Server: ComputeServer,
    Server::Info: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("ServerUtilities")
            .field("service", &self.service)
            .field("properties", &self.properties)
            .field("info", &self.info)
            .field("logger", &self.logger)
            .finish()
    }
}

impl<S: ComputeServer> ServerUtilities<S> {
    /// Creates a new server utilities.
    pub fn new(
        service: ServiceId,
        properties: DeviceProperties,
        logger: Arc<ServerLogger>,
        info: S::Info,
        allocator: S::MemoryLayoutPolicy,
    ) -> Self {
        // Start a tracy client if needed.
        #[cfg(feature = "runtime-profile-tracy")]
        let client = tracy_client::Client::start();

        Self {
            service,
            properties_hash: properties.checksum(),
            properties,
            logger,
            // Create the GPU client if needed.
            #[cfg(feature = "runtime-profile-tracy")]
            gpu_client: client
                .clone()
                .new_gpu_context(
                    Some(&format!("{info:?}")),
                    // In the future should ask the server what makes sense here. 'Invalid' atm is a generic stand-in (Tracy doesn't have CUDA/RocM atm anyway).
                    tracy_client::GpuContextType::Invalid,
                    0,   // Timestamps are manually aligned to this epoch so start at 0.
                    1.0, // Timestamps are manually converted to be nanoseconds so period is 1.
                )
                .unwrap(),
            #[cfg(feature = "runtime-profile-tracy")]
            epoch_time: web_time::Instant::now(),
            info,
            runtime_name: None,
            layout_policy: allocator,
            check_mode: RudaRuntimeConfig::get().compilation.check_mode,
            initialized_comms: RwLock::new(HashSet::default()),
        }
    }
}

/// Kernel Launch Errors.
#[derive(Error, Clone)]
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
pub enum LaunchError {
    /// The given kernel can't be compiled.
    #[error("A compilation error happened during launch\nCaused by:\n  {0}")]
    CompilationError(#[from] CompilationError),

    /// The server is out of memory.
    #[error(
        "An out-of-memory error happened during launch\nCaused by:\n  {reason}\nBacktrace\n{backtrace}"
    )]
    OutOfMemory {
        /// The caused of the memory error.
        reason: String,
        /// The backtrace for this error.
        #[cfg_attr(serializable, serde(skip))]
        backtrace: BackTrace,
    },

    /// Too many resources were requested
    #[error("Too many resources were requested during launch\n{0}")]
    TooManyResources(#[from] ResourceLimitError),

    /// Unknown launch error.
    #[error(
        "An unknown error happened during launch\nCaused by:\n  {reason}\nBacktrace\n{backtrace}"
    )]
    Unknown {
        /// The caused of the unknown error.
        reason: String,
        /// The backtrace for this error.
        #[cfg_attr(serializable, serde(skip))]
        backtrace: BackTrace,
    },

    /// Can't launch because of an IO Error.
    #[error("An io error happened during launch\nCaused by:\n  {0}")]
    IoError(#[from] IoError),
    /// The driver reports an unrecoverable context/device fault.
    #[error("The device was poisoned during launch\n{0}")]
    DevicePoisoned(#[from] DevicePoison),
}

impl LaunchError {
    /// Whether the driver can no longer execute work on the device.
    pub fn is_device_poisoned(&self) -> bool {
        match self {
            Self::DevicePoisoned(_) => true,
            Self::CompilationError(error) => error.is_device_poisoned(),
            Self::IoError(error) => error.is_device_poisoned(),
            _ => false,
        }
    }

    /// Whether the backend declined a kernel configuration before running it.
    pub fn is_refusal(&self) -> bool {
        match self {
            Self::CompilationError(error) => error.is_refusal(),
            Self::IoError(error) => error.is_refusal(),
            Self::TooManyResources(_) => true,
            _ => false,
        }
    }
}

/// Resource limit errors.
#[derive(Error, Clone)]
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
pub enum ResourceLimitError {
    /// Shared memory exceeds maximum
    #[error(
        "Too much shared memory requested.\nRequested {requested} bytes, maximum {max} bytes available.\nBacktrace\n{backtrace}"
    )]
    SharedMemory {
        /// Value requested
        requested: usize,
        /// Maximum value
        max: usize,
        /// The backtrace for this error.
        #[cfg_attr(serializable, serde(skip))]
        backtrace: BackTrace,
    },
    /// Total units exceeds maximum
    #[error(
        "Total unit count exceeds maximum.\nRequested {requested} units, max units is {max}.\nBacktrace\n{backtrace}"
    )]
    Units {
        /// Requested value
        requested: u32,
        /// Maximum value
        max: u32,
        /// The backtrace for this error.
        #[cfg_attr(serializable, serde(skip))]
        backtrace: BackTrace,
    },
    /// `RudaDim` exceeds maximum
    #[error(
        "Ruda dim exceeds maximum bounds.\nRequested {requested:?}, max is {max:?}.\nBacktrace\n{backtrace}"
    )]
    RudaDim {
        /// Requested value
        requested: (u32, u32, u32),
        /// Maximum value
        max: (u32, u32, u32),
        /// The backtrace for this error.
        #[cfg_attr(serializable, serde(skip))]
        backtrace: BackTrace,
    },
}

impl core::fmt::Debug for LaunchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self}"))
    }
}

impl core::fmt::Debug for ResourceLimitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self}"))
    }
}

/// An operation using a runtime's native inter-device transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
pub enum Collective {
    /// Initialize a group of devices.
    CommInit,
    /// Reduce a buffer across a group.
    AllReduce,
    /// Send a buffer to another device.
    Send,
    /// Receive a buffer from another device.
    Recv,
    /// Wait for queued collective work.
    SyncCollective,
}

impl core::fmt::Display for Collective {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::CommInit => "comm_init",
            Self::AllReduce => "all_reduce",
            Self::Send => "send",
            Self::Recv => "recv",
            Self::SyncCollective => "sync_collective",
        })
    }
}

/// Error that can happen asynchronously while executing registered kernels.
#[derive(Error, Debug, Clone)]
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
pub enum ServerError {
    /// A generic runtime error.
    #[error("An error happened during execution\nCaused by:\n  {reason}\nBacktrace:\n{backtrace}")]
    Generic {
        /// The details of the generic error.
        reason: String,
        /// The backtrace for this error.
        #[cfg_attr(serializable, serde(skip))]
        backtrace: BackTrace,
    },

    /// A launch error happened during profiling
    #[error("A launch error happened during profiling\nCaused by:\n  {0}")]
    Launch(#[from] LaunchError),

    /// An execution error happened during profiling
    #[error("An execution error happened during profiling\nCaused by:\n  {0}")]
    Profile(#[from] ProfileError),

    /// An execution error happened during profiling
    #[error("An execution error happened during profiling\nCaused by:\n  {0}")]
    Io(#[from] IoError),

    /// The server is an invalid state.
    #[error("The server is in an invalid state\nCaused by:\n  {errors:?}")]
    ServerUnhealthy {
        /// The details of the generic error.
        errors: Vec<Self>,
        /// The backtrace for this error.
        #[cfg_attr(serializable, serde(skip))]
        backtrace: BackTrace,
    },

    /// A root failure and the dependency path through buffers it left unwritten.
    #[error("The bytes were never written (failure #{failure}, claiming {claimed} buffers)\n{chain:?}\nCaused by:\n{root}\n{backtrace}")]
    Unwritten {
        failure: u64,
        claimed: u32,
        chain: Vec<String>,
        root: Box<ServerError>,
        #[cfg_attr(serializable, serde(skip))]
        backtrace: BackTrace,
    },

    /// Distinct failures carried by a set of buffers.
    #[error("Several failures at once\nCaused by:\n{errors:?}\n{backtrace}")]
    Several {
        errors: Vec<Self>,
        #[cfg_attr(serializable, serde(skip))]
        backtrace: BackTrace,
    },
    /// A producer was dropped without completing or reporting its failure.
    #[error("Producer was torn down before its write scope completed")]
    TornDown,
    /// The underlying driver/device can no longer execute work.
    #[error("The device is poisoned\n{0}")]
    DevicePoisoned(#[from] DevicePoison),
    /// A handle was supplied to a different device service; no work was submitted.
    #[error("A handle of {handle} was used on {client}\nBacktrace:\n{backtrace}")]
    ForeignHandle {
        /// The service that owns the handle.
        handle: String,
        /// The service reached by the client.
        client: String,
        #[cfg_attr(serializable, serde(skip))]
        backtrace: BackTrace,
    },
    /// A native resource was requested through the wrong concrete service type.
    #[error("The client reaches {client}, not a {requested}\nBacktrace:\n{backtrace}")]
    ServiceMismatch {
        /// The service reached by the client.
        client: String,
        /// The concrete service type requested by the caller.
        requested: String,
        #[cfg_attr(serializable, serde(skip))]
        backtrace: BackTrace,
    },
    /// The runtime has no native inter-device transport for this operation.
    #[error("{operation} needs a transport between devices, and this runtime has none\nBacktrace:\n{backtrace}")]
    NoDeviceTransport {
        /// The requested operation.
        operation: Collective,
        #[cfg_attr(serializable, serde(skip))]
        backtrace: BackTrace,
    },
}

impl ServerError {
    /// Report an unsupported native collective without touching its buffers.
    pub fn no_device_transport(operation: Collective) -> Self {
        Self::NoDeviceTransport { operation, backtrace: BackTrace::capture() }
    }

    /// Whether any root failure reports an unrecoverable device fault.
    pub fn is_device_poisoned(&self) -> bool {
        match self {
            Self::DevicePoisoned(_) => true,
            Self::Launch(error) => error.is_device_poisoned(),
            Self::Io(error) => error.is_device_poisoned(),
            Self::Profile(error) => error.is_device_poisoned(),
            Self::Unwritten { root, .. } => root.is_device_poisoned(),
            Self::Several { errors, .. } | Self::ServerUnhealthy { errors, .. } =>
                errors.iter().any(Self::is_device_poisoned),
            _ => false,
        }
    }

    /// Whether every reported root is a pre-execution kernel refusal.
    pub fn is_refusal(&self) -> bool {
        match self {
            Self::Launch(error) => error.is_refusal(),
            Self::Io(error) => error.is_refusal(),
            Self::Profile(error) => error.is_refusal(),
            Self::Unwritten { root, .. } => root.is_refusal(),
            Self::Several { errors, .. } | Self::ServerUnhealthy { errors, .. } =>
                !errors.is_empty() && errors.iter().all(Self::is_refusal),
            _ => false,
        }
    }
}

/// How errors are handled in a stream when executing a task.
#[derive(Clone, Copy)]
pub struct StreamErrorMode {
    /// Whether the task still executes even if the stream is in error.
    pub ignore: bool,
    /// Whether the errors are flushed by the current task.
    pub flush: bool,
}

/// The compute server is responsible for handling resources and computations over resources.
///
/// Everything in the server is mutable, therefore it should be solely accessed through the
/// [`ComputeClient`] for thread safety.
pub trait ComputeServer:
    Send + core::fmt::Debug + ServerCommunication + device::DeviceService + 'static
where
    Self: Sized,
{
    /// The kernel type defines the computation algorithms.
    type Kernel: KernelMetadata;
    /// Information that can be retrieved for the runtime.
    type Info: Debug + Send + Sync;
    /// Manages how allocations are performed for a server.
    type MemoryLayoutPolicy: MemoryLayoutPolicy;
    /// The [storage](ComputeStorage) type defines how data is stored and accessed.
    type Storage: ComputeStorage;

    /// Initializes [memory](ManagedMemoryHandle) on the given [stream](StreamId) with the given size.
    fn initialize_memory(&mut self, memory: ManagedMemoryHandle, size: u64, stream_id: StreamId);

    /// Reserves N [Bytes] of the provided sizes to be used as staging to load data.
    fn staging(
        &mut self,
        _sizes: &[usize],
        _stream_id: StreamId,
    ) -> Result<Vec<Bytes>, ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    /// Retrieve the server logger.
    fn logger(&self) -> Arc<ServerLogger>;

    /// Retrieve the server utilities.
    fn utilities(&self) -> Arc<ServerUtilities<Self>>;

    /// Given bindings, returns the owned resources as bytes.
    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>>;

    /// Writes the specified bytes into the buffers given
    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, stream_id: StreamId);

    /// Wait for the completion of every task in the server.
    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ServerError>>;

    /// Check buffer claims without submitting a device barrier or copying bytes.
    fn check(&mut self, bindings: Vec<Binding>, _stream_id: StreamId) -> Result<(), ServerError> {
        Binding::check_all(bindings.iter())
    }

    /// Complete the issuing and owning streams, then check the requested buffers.
    fn sync_buffers(
        &mut self,
        bindings: Vec<Binding>,
        stream_id: StreamId,
    ) -> DynFut<Result<(), ServerError>> {
        let mut streams = alloc::vec![stream_id];
        for binding in &bindings {
            if !streams.contains(&binding.stream) {
                streams.push(binding.stream);
            }
        }
        let futures: Vec<_> = streams.into_iter().map(|stream| self.sync(stream)).collect();
        Box::pin(async move {
            let mut first_error = None;
            for future in futures {
                if let Err(error) = future.await {
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
            }
            if let Some(error) = first_error {
                return Err(error);
            }
            Binding::check_all(bindings.iter())
        })
    }

    /// Given a resource handle, returns the storage resource.
    fn get_resource(
        &mut self,
        binding: Binding,
        stream_id: StreamId,
    ) -> Result<ManagedResource<<Self::Storage as ComputeStorage>::Resource>, ServerError>;

    /// Executes the `kernel` over the given memory `handles`.
    ///
    /// Kernels have mutable access to every resource they are given
    /// and are responsible of determining which should be read or written.
    ///
    /// # Safety
    ///
    /// When executing with mode [`ExecutionMode::Unchecked`], out-of-bound reads and writes can happen.
    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: RudaCount,
        bindings: KernelArguments,
        kind: ExecutionMode,
        stream_id: StreamId,
    );

    /// Compile the kernel and either dispatch it or retain only its compiled artifact.
    /// The mode is captured on the issuing thread, before submission to the server.
    ///
    /// Backends supporting dry runs must override this method. The default preserves
    /// normal launches for existing backends without silently executing skipped work.
    ///
    /// # Safety
    /// The same requirements as [`Self::launch`] apply when executing the kernel.
    unsafe fn launch_with_mode(
        &mut self,
        kernel: Self::Kernel,
        count: RudaCount,
        bindings: KernelArguments,
        kind: ExecutionMode,
        stream_id: StreamId,
        launch_mode: crate::runtime::dry_run::LaunchMode,
    ) {
        assert!(!launch_mode.is_skipped(), "This compute server does not implement dry-run compilation");
        unsafe { self.launch(kernel, count, bindings, kind, stream_id) };
    }

    /// Flush all outstanding tasks in the server.
    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError>;

    /// Prepare pools and metadata for a closure's graph-capture warmup.
    fn graph_prepare(&mut self, _stream: StreamId) -> Result<(), ServerError> {
        Err(ServerError::Generic { reason: "Hardware graph capture is unsupported".into(), backtrace: BackTrace::capture() })
    }

    /// Open a recording window on the selected stream.
    fn graph_start_capture(&mut self, _stream: StreamId) -> Result<(), ServerError> {
        Err(ServerError::Generic { reason: "Hardware graph capture is unsupported".into(), backtrace: BackTrace::capture() })
    }

    /// Close the recording even on failure and return a server-owned graph id.
    fn graph_stop_capture(&mut self, _stream: StreamId) -> Result<u64, ServerError> {
        Err(ServerError::Generic { reason: "Hardware graph capture is unsupported".into(), backtrace: BackTrace::capture() })
    }

    /// Close a recording that unwound without producing an output.
    fn graph_abort_capture(&mut self, _stream: StreamId) {}

    /// Enqueue a captured graph on its original stream.
    ///
    /// # Safety
    /// Captured inputs must remain valid and must not race another stream.
    unsafe fn graph_replay(&mut self, _stream: StreamId, _graph: u64) -> Result<(), ServerError> {
        Err(ServerError::Generic { reason: "Hardware graph replay is unsupported".into(), backtrace: BackTrace::capture() })
    }

    /// Release the executable and its retained pages after pending work completes.
    fn graph_release(&mut self, _stream: StreamId, _graph: u64) {}

    /// The current memory usage of the server.
    fn memory_usage(&mut self, stream_id: StreamId) -> Result<MemoryUsage, ServerError>;

    /// Ask the server to release memory that it can release.
    fn memory_cleanup(&mut self, stream_id: StreamId);

    /// Explicitly compact eligible allocations and release empty pages.
    /// Backends must submit pending address users and confirm relocation completion.
    fn memory_compact(&mut self, _stream_id: StreamId) -> Result<(), ServerError> {
        Err(IoError::UnsupportedIoOperation { backtrace: BackTrace::capture() }.into())
    }

    /// Enable collecting timestamps.
    fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError>;

    /// Disable collecting timestamps.
    fn end_profile(
        &mut self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError>;

    /// Release a window that will not be measured. Native implementations need not flush.
    fn abandon_profile(&mut self, stream_id: StreamId, token: ProfilingToken) {
        let _ = self.end_profile(stream_id, token);
    }

    /// Update the memory mode of allocation in the server.
    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId);
}

pub use ruda_core::device::CommunicationId;

/// Different reduce operations.
pub enum ReduceOperation {
    /// Sum.
    Sum,
    /// Mean.
    Mean,
}

/// Defines functions for optimized data transfer between servers, supporting custom communication
/// mechanisms such as peer-to-peer communication or specialized implementations.
pub trait ServerCommunication {
    /// Indicates whether server-to-server communication is enabled for this implementation.
    const SERVER_COMM_ENABLED: bool;

    /// Ensure that all queued collective operations have been executed.
    ///
    /// # Arguments
    ///
    /// * `stream_id` - The [`StreamId`] of the stream waiting for the sync.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing an `ServerError` if the operation fails.
    #[allow(unused_variables)]
    fn sync_collective(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        Err(ServerError::no_device_transport(Collective::SyncCollective))
    }

    /// Initialize the communication between the devices in `device_ids`.
    ///
    /// # Arguments
    ///
    /// * `device_ids` - The IDs of the devices that need communication.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing an `ServerError` if the operation fails.
    #[allow(unused_variables)]
    fn comm_init(&mut self, device_ids: Vec<DeviceId>) -> Result<(), ServerError> {
        Err(ServerError::no_device_transport(Collective::CommInit))
    }

    /// Performs an `all_reduce` operation on the input data and writes it to the output buffer.
    /// see <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce>
    ///
    /// # Arguments
    ///
    /// * `src` - The data to be reduced.
    /// * `dst` - Where to write the result.
    /// * `dtype` - The element type of the data being reduced
    /// * `stream_id` - The data's stream id.
    /// * `op` - The reduce's aggregation operation e.g. mean, sum, etc.
    /// * `device_ids` - The list of device ids from which to `all_reduce`.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing an `ServerError` if the operation fails.
    #[allow(unused_variables)]
    fn all_reduce(
        &mut self,
        src: Binding,
        dst: Binding,
        dtype: ElemType,
        stream_id: StreamId,
        op: ReduceOperation,
        device_ids: Vec<DeviceId>,
    ) -> Result<(), ServerError> {
        Err(ServerError::no_device_transport(Collective::AllReduce))
    }

    /// Sends data from this server to a destination server.
    ///
    /// # Arguments
    ///
    /// * `desc` - A descriptor specifying the data to be sent, including shape, strides, and binding.
    /// * `dtype` - The element type of the data being sent.
    /// * `stream_id` - The stream ID associated with the server's operation.
    /// * `device_id_dst` - ID of the device receiving the data.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing an `ServerError` if the operation fails.
    #[allow(unused_variables)]
    fn send(
        &mut self,
        desc: CopyDescriptor,
        dtype: ElemType,
        stream_id: StreamId,
        device_id_dst: DeviceId,
    ) -> Result<(), ServerError> {
        Err(ServerError::no_device_transport(Collective::Send))
    }

    /// Receive data from another server.
    ///
    /// # Arguments
    ///
    /// * `handle` - The handle in which the received data is written.
    /// * `dtype` - The element type of the data being sent.
    /// * `stream_id` - The stream ID associated with the server's operation.
    /// * `device_id_src` - ID of the device sending the data.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing an `ServerError` if the operation fails.
    #[allow(unused_variables)]
    fn recv(
        &mut self,
        handle: Handle,
        dtype: ElemType,
        stream_id: StreamId,
        device_id_src: DeviceId,
    ) -> Result<(), ServerError> {
        Err(ServerError::no_device_transport(Collective::Recv))
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
/// Profiling identification so that the server can support recursive and overlapping profilings.
pub struct ProfilingToken {
    /// The token value.
    pub id: u64,
}

/// Type of allocation, either contiguous or optimized (row-aligned when possible)
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum MemoryLayoutStrategy {
    /// Contiguous layout, with no padding
    Contiguous,
    /// Optimized for access speed. In practice this means row-aligned with padding for runtimes
    /// that support it.
    Optimized,
}

/// Descriptor for a new tensor allocation
#[derive(new, Debug, Clone)]
pub struct MemoryLayoutDescriptor {
    /// Strategy used to create the memory layout.
    pub strategy: MemoryLayoutStrategy,
    /// Shape of the tensor
    pub shape: Shape,
    /// Size of each element in the tensor (used for conversion of shape to bytes)
    pub elem_size: usize,
}

impl MemoryLayoutDescriptor {
    /// Create an optimized allocation descriptor
    pub fn optimized(shape: Shape, elem_size: usize) -> Self {
        MemoryLayoutDescriptor::new(MemoryLayoutStrategy::Optimized, shape, elem_size)
    }

    /// Create a contiguous allocation descriptor
    pub fn contiguous(shape: Shape, elem_size: usize) -> Self {
        MemoryLayoutDescriptor::new(MemoryLayoutStrategy::Contiguous, shape, elem_size)
    }
}

/// An allocation with associated strides. Strides depend on tensor layout.
#[derive(Debug, Clone)]
pub struct MemoryLayout {
    /// The handle for the memory resource
    pub memory: Handle,
    /// TODO: `Strides` should become `Layout`.
    ///
    /// The strides of the tensor
    pub strides: Strides,
}

impl MemoryLayout {
    /// Create a new memory layout.
    pub fn new(handle: Handle, strides: impl Into<Strides>) -> Self {
        MemoryLayout {
            memory: handle,
            strides: strides.into(),
        }
    }
}

/// A reason for an error.
#[derive(Default, Clone)]
pub struct Reason {
    inner: ReasonInner,
}

#[cfg(std_io)]
mod _reason_serde {
    use super::*;

    use alloc::string::ToString;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    impl Serialize for Reason {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            // Use the Display implementation (via to_string) to flatten the enum
            serializer.serialize_str(&self.to_string())
        }
    }

    impl<'de> Deserialize<'de> for Reason {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            // Deserialize into a standard String first
            let s = String::deserialize(deserializer)?;

            // Wrap it in the Dynamic variant since we can't safely
            // reconstruct a 'static str from a runtime string.
            Ok(Reason {
                inner: ReasonInner::Dynamic(Arc::new(s)),
            })
        }
    }
}

#[derive(Default, Clone)]
enum ReasonInner {
    Static(&'static str),
    Dynamic(Arc<String>),
    #[default]
    NotProvided,
}

impl core::fmt::Display for Reason {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match &self.inner {
            ReasonInner::Static(content) => f.write_str(content),
            ReasonInner::Dynamic(content) => f.write_str(content),
            ReasonInner::NotProvided => f.write_str("No reason provided for the error"),
        }
    }
}

impl core::fmt::Debug for Reason {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(&self, f)
    }
}

impl From<&'static str> for Reason {
    fn from(value: &'static str) -> Self {
        Self {
            inner: ReasonInner::Static(value),
        }
    }
}

impl From<String> for Reason {
    fn from(value: String) -> Self {
        Self {
            inner: ReasonInner::Dynamic(Arc::new(value)),
        }
    }
}

/// Error returned from `create`/`read`/`write` functions. Due to async execution not all errors
/// are able to be caught, so some IO errors will still panic.
#[derive(Error, Clone)]
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
pub enum IoError {
    /// Buffer size exceeds the max available
    #[error("can't allocate buffer of size: {size}\n{backtrace}")]
    BufferTooBig {
        /// The size of the buffer in bytes.
        size: u64,
        /// The captured backtrace.
        #[cfg_attr(serializable, serde(skip))]
        backtrace: BackTrace,
    },

    /// Strides aren't supported for this copy operation on this runtime
    #[error("the provided strides are not supported for this operation\n{backtrace}")]
    UnsupportedStrides {
        /// The backtrace.
        #[cfg_attr(serializable, serde(skip))]
        backtrace: BackTrace,
    },

    /// Memory wasn't found in the memory pool
    #[error("couldn't find resource for that handle: {reason}\n{backtrace}")]
    NotFound {
        /// The backtrace.
        #[cfg_attr(serializable, serde(skip))]
        backtrace: BackTrace,
        /// The reason the handle is invalid.
        reason: Reason,
    },

    /// Handle wasn't found in the memory pool
    #[error("couldn't free the handle, since it is currently in used. \n{backtrace}")]
    FreeError {
        /// The backtrace.
        #[cfg_attr(serializable, serde(skip))]
        backtrace: BackTrace,
    },

    /// Unknown error happened during execution
    #[error("Unknown error happened during execution\n{backtrace}")]
    Unknown {
        /// Details of the error
        description: String,
        /// The backtrace.
        #[cfg_attr(serializable, serde(skip))]
        backtrace: BackTrace,
    },

    /// The current IO operation is not supported
    #[error("The current IO operation is not supported\n{backtrace}")]
    UnsupportedIoOperation {
        /// The backtrace.
        #[cfg_attr(serializable, serde(skip))]
        backtrace: BackTrace,
    },

    /// Can't perform the IO operation because of a runtime error.
    #[error("Can't perform the IO operation because of a runtime error: {0}")]
    Execution(#[from] Box<ServerError>),
    /// The driver reports an unrecoverable device/context fault.
    #[error("The device is poisoned during IO\n{0}")]
    DevicePoisoned(#[from] DevicePoison),
}

impl IoError {
    /// Preserve the classification of a runtime failure wrapped by an IO operation.
    pub fn is_refusal(&self) -> bool {
        match self {
            Self::Execution(error) => error.is_refusal(),
            _ => false,
        }
    }

    /// Whether this IO operation failed because the device is unusable.
    pub fn is_device_poisoned(&self) -> bool {
        match self {
            Self::DevicePoisoned(_) => true,
            Self::Execution(error) => error.is_device_poisoned(),
            _ => false,
        }
    }
}

impl core::fmt::Debug for IoError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self}"))
    }
}

/// Arguments to execute a kernel.
#[derive(Debug, Default)]
pub struct KernelArguments {
    /// Buffer bindings
    pub buffers: Vec<Binding>,
    /// Launch-site accesses, used only where the compiled kernel has no answer.
    /// Missing entries are read-write; buffer and tensor-map indices remain separate.
    pub declared_io: ruda_core::kernel::KernelIo,
    /// Packed scalars and metadata. First scalars sorted by type, then static metadata,
    /// then dynamic metadata.
    pub info: MetadataBindingInfo,
    /// Tensor map bindings
    pub tensor_maps: Vec<TensorMapBinding>,
}

impl core::fmt::Display for KernelArguments {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("KernelArguments")?;
        for b in self.buffers.iter() {
            f.write_fmt(format_args!("\n - buffer: {b:?}\n"))?;
        }

        Ok(())
    }
}

impl KernelArguments {
    /// Create a new bindings struct
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a buffer binding
    pub fn with_buffer(mut self, binding: Binding) -> Self {
        self.buffers.push(binding);
        self
    }

    /// Add a buffer with a fallback declaration that survives compilation failure.
    pub fn with_buffer_io(mut self, binding: Binding, io: ruda_core::kernel::BufferIOAttr) -> Self {
        self.declared_io.buffers.resize(self.buffers.len(), ruda_core::kernel::BufferIOAttr::ReadWrite);
        self.buffers.push(binding);
        self.declared_io.buffers.push(io);
        self
    }

    /// Extend the buffers with `bindings`
    pub fn with_buffers(mut self, bindings: Vec<Binding>) -> Self {
        self.buffers.extend(bindings);
        self
    }

    /// Set the info to `info`
    pub fn with_info(mut self, info: MetadataBindingInfo) -> Self {
        self.info = info;
        self
    }

    /// Extend the tensor maps with `bindings`
    pub fn with_tensor_maps(mut self, bindings: Vec<TensorMapBinding>) -> Self {
        self.tensor_maps.extend(bindings);
        self
    }

    /// Add a tensor map with an explicit fallback access declaration.
    pub fn with_tensor_map_io(mut self, binding: TensorMapBinding, io: ruda_core::kernel::BufferIOAttr) -> Self {
        self.declared_io.tensor_maps.resize(self.tensor_maps.len(), ruda_core::kernel::BufferIOAttr::ReadWrite);
        self.tensor_maps.push(binding);
        self.declared_io.tensor_maps.push(io);
        self
    }

    /// Payloads the launch may read: compiled access, then declaration, then read-write.
    pub fn buffers_read<'a>(&'a self, io: Option<&'a ruda_core::kernel::KernelIo>)
        -> impl Iterator<Item = &'a Binding>
    {
        self.bindings_with_io(io).filter_map(|(binding, access)| access.reads().then_some(binding))
    }

    /// Payloads the launch may write, including tensor-map backing buffers.
    pub fn buffers_written<'a>(&'a self, io: Option<&'a ruda_core::kernel::KernelIo>)
        -> impl Iterator<Item = &'a Binding>
    {
        self.bindings_with_io(io).filter_map(|(binding, access)| access.writes().then_some(binding))
    }

    fn bindings_with_io<'a>(&'a self, io: Option<&'a ruda_core::kernel::KernelIo>)
        -> impl Iterator<Item = (&'a Binding, ruda_core::kernel::BufferIOAttr)>
    {
        use ruda_core::kernel::BufferIOAttr;
        self.buffers.iter().enumerate().map(move |(index, binding)| {
            let access = io.and_then(|io| io.buffers.get(index))
                .or_else(|| self.declared_io.buffers.get(index))
                .copied().unwrap_or(BufferIOAttr::ReadWrite);
            (binding, access)
        }).chain(self.tensor_maps.iter().enumerate().map(move |(index, map)| {
            let access = io.and_then(|io| io.tensor_maps.get(index))
                .or_else(|| self.declared_io.tensor_maps.get(index))
                .copied().unwrap_or(BufferIOAttr::ReadWrite);
            (&map.binding, access)
        }))
    }
}

/// Binding of a set of scalars of the same type to execute a kernel.
///
/// The [`ComputeServer`] is responsible to convert those info into actual [`Binding`] when launching
/// kernels.
#[derive(new, Debug, Default)]
pub struct MetadataBindingInfo {
    /// Scalar and metadata values
    pub data: Vec<u64>,
    /// Start of the dynamically sized portion of the metadata, relative to the entire info buffer
    pub dynamic_metadata_offset: usize,
}

impl MetadataBindingInfo {
    /// Create a new binding info for custom data, for externally compiled kernels.
    pub fn custom(data: Vec<u64>) -> Self {
        Self::new(data, 0)
    }
}

/// A binding with shape and stride info for non-contiguous reading
#[derive(new, Debug)]
pub struct CopyDescriptor {
    /// Binding for the memory resource
    pub handle: Binding,
    /// Shape of the resource
    pub shape: Shape,
    /// Strides of the resource
    pub strides: Strides,
    /// Size of each element in the resource
    pub elem_size: usize,
}

/// A tensor map used with TMA ops
#[derive(new, Debug)]
pub struct TensorMapBinding {
    /// The binding for the backing tensor
    pub binding: Binding,
    /// The tensormap metadata
    pub map: TensorMapMeta,
}

/// `TensorMap` metadata for the opaque proxy used in TMA copies
#[derive(Debug, Clone)]
pub struct TensorMapMeta {
    /// Tensormap format (tiled or im2col)
    pub format: TensorMapFormat,
    /// Metadata of the backing tensor
    pub metadata: Metadata,
    /// Element stride, usually 1 but may be 2 for complex tensors
    /// For im2col, this is equivalent to the kernel stride
    pub elem_stride: Strides,
    /// Interleave mode
    pub interleave: TensorMapInterleave,
    /// Swizzle mode
    pub swizzle: TensorMapSwizzle,
    /// Prefetch settings
    pub prefetch: TensorMapPrefetch,
    /// OOB fill value
    pub oob_fill: OobFill,
    /// Storage type
    pub storage_ty: StorageType,
}

/// Specifieds the number of rudas to be dispatched for a kernel.
///
/// This translates to eg. a grid for CUDA, or to `num_workgroups` for wgsl.
#[allow(clippy::large_enum_variant)]
pub enum RudaCount {
    /// Dispatch a known count of x, y, z rudas.
    Static(u32, u32, u32),
    /// Dispatch an amount based on the values in this buffer. The buffer should contain a u32 array [x, y, z].
    Dynamic(Binding),
}

/// Defines how to select ruda count based on the number of rudas required.
pub enum RudaCountSelection {
    /// If the number of rudas is the same as required.
    Exact(RudaCount),
    /// If the number of rudas isn't the same as required.
    ///
    /// This can happen based on the hardware limit, requiring the kernel to perform OOB checks.
    Approx(RudaCount, u32),
}

impl RudaCountSelection {
    /// Creates a [`RudaCount`] while respecting the hardware limits.
    pub fn new<R: Runtime>(client: &ComputeClient<R>, num_rudas: u32) -> Self {
        let ruda_count = ruda_count_spread(&client.properties().hardware.max_ruda_count, num_rudas);

        let num_rudas_actual = ruda_count[0] * ruda_count[1] * ruda_count[2];
        let ruda_count = RudaCount::Static(ruda_count[0], ruda_count[1], ruda_count[2]);

        match num_rudas_actual == num_rudas {
            true => RudaCountSelection::Exact(ruda_count),
            false => RudaCountSelection::Approx(ruda_count, num_rudas_actual),
        }
    }

    /// If some rudas will be idle.
    pub fn has_idle(&self) -> bool {
        matches!(self, Self::Approx(..))
    }

    /// Converts into [`RudaCount`].
    pub fn ruda_count(self) -> RudaCount {
        match self {
            RudaCountSelection::Exact(ruda_count) => ruda_count,
            RudaCountSelection::Approx(ruda_count, _) => ruda_count,
        }
    }
}

impl From<RudaCountSelection> for RudaCount {
    fn from(value: RudaCountSelection) -> Self {
        value.ruda_count()
    }
}

impl RudaCount {
    /// Create a new static ruda count with the given x = y = z = 1.
    pub fn new_single() -> Self {
        RudaCount::Static(1, 1, 1)
    }

    /// Create a new static ruda count with the given x, and y = z = 1.
    pub fn new_1d(x: u32) -> Self {
        RudaCount::Static(x, 1, 1)
    }

    /// Create a new static ruda count with the given x and y, and z = 1.
    pub fn new_2d(x: u32, y: u32) -> Self {
        RudaCount::Static(x, y, 1)
    }

    /// Create a new static ruda count with the given x, y and z.
    pub fn new_3d(x: u32, y: u32, z: u32) -> Self {
        RudaCount::Static(x, y, z)
    }

    /// Checks whether the ruda count is definitely empty, i.e. has 0 dispatches.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Static(x, y, z) => *x == 0 || *y == 0 || *z == 0,
            Self::Dynamic(_) => false,
        }
    }
}

impl Debug for RudaCount {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RudaCount::Static(x, y, z) => f.write_fmt(format_args!("({x}, {y}, {z})")),
            RudaCount::Dynamic(_) => f.write_str("binding"),
        }
    }
}

impl Clone for RudaCount {
    fn clone(&self) -> Self {
        match self {
            Self::Static(x, y, z) => Self::Static(*x, *y, *z),
            Self::Dynamic(binding) => Self::Dynamic(binding.clone()),
        }
    }
}

pub use ruda_core::launch::{RudaDim, ExecutionMode};

fn ruda_count_spread(max: &(u32, u32, u32), num_rudas: u32) -> [u32; 3] {
    let max_ruda_counts = [max.0, max.1, max.2];
    let mut num_rudas = [num_rudas, 1, 1];
    let base = 2;

    let mut reduce_count = |i: usize| {
        if num_rudas[i] <= max_ruda_counts[i] {
            return true;
        }

        loop {
            num_rudas[i] = num_rudas[i].div_ceil(base);
            num_rudas[i + 1] *= base;

            if num_rudas[i] <= max_ruda_counts[i] {
                return false;
            }
        }
    };

    for i in 0..2 {
        if reduce_count(i) {
            break;
        }
    }

    num_rudas
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_log::test]
    fn safe_num_rudas_even() {
        let max = (32, 32, 32);
        let required = 2048;

        let actual = ruda_count_spread(&max, required);
        let expected = [32, 32, 2];
        assert_eq!(actual, expected);
    }

    #[test_log::test]
    fn safe_num_rudas_odd() {
        let max = (48, 32, 16);
        let required = 3177;

        let actual = ruda_count_spread(&max, required);
        let expected = [25, 32, 4];
        assert_eq!(actual, expected);
    }
}
