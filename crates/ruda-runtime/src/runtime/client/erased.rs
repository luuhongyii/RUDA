use super::{ComputeClient, ProfileWindow};
use crate::runtime::{
    backend::{DeviceUtilization, Runtime, UtilizationUnavailable},
    capture::RuntimeGraph,
    kernel::{KernelTask, RudaKernel},
    memory_management::{MemoryAllocationMode, MemoryUsage},
    server::{
        ComputeServer, CopyDescriptor, Handle, IoError, KernelArguments, MemoryLayout,
        MemoryLayoutDescriptor, MemoryLayoutStrategy, ProfileError, ReduceOperation, RudaCount,
        ServerCommunication, ServerError,
    },
    storage::{ComputeStorage, ManagedResource},
    throughput::{ThroughputError, ThroughputKey, ThroughputValue},
};
use alloc::{boxed::Box, sync::Arc, vec, vec::Vec};
use core::any::{Any, TypeId};
use ruda_core::{
    backtrace::BackTrace,
    bytes::Bytes,
    device::{Device, DeviceId, ServiceId},
    future::DynFut,
    ir::{DeviceProperties, ElemType, TargetProperties, VectorSize, features::Features},
    profile::ProfileDuration,
    stream_id::StreamId,
    tensor::{Shape, Strides},
};

type Value = Box<dyn Any + Send>;
type Task<'a> = Box<dyn FnOnce() -> Value + Send + 'a>;

/// A client whose runtime is selected at run time.
///
/// Erasure retains the original service, allocator, compilation cache and stream selection.
pub struct ErasedClient {
    backend: Box<dyn ClientBackend>,
}

/// The runtime-independent compute client.
pub type Client = ErasedClient;

impl Clone for ErasedClient {
    fn clone(&self) -> Self { Self { backend: self.backend.clone_box() } }
}

impl core::fmt::Debug for ErasedClient {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Client").field("name", &self.name())
            .field("service", &self.service_id()).field("stream", &self.execution_stream()).finish()
    }
}

impl<R: Runtime> From<ComputeClient<R>> for ErasedClient {
    fn from(client: ComputeClient<R>) -> Self { Self { backend: Box::new(client) } }
}

/// A captured graph retaining its original runtime and capture stream.
#[derive(Clone)]
pub struct Graph {
    inner: Arc<dyn GraphBackend>,
}

impl Graph {
    /// Enqueue the recorded work on its capture stream.
    ///
    /// # Safety
    /// All captured buffers must remain live and have no concurrent accesses.
    pub unsafe fn replay(&self) -> Result<(), ServerError> {
        unsafe { self.inner.replay() }
    }
}

trait GraphBackend: Send + Sync {
    unsafe fn replay(&self) -> Result<(), ServerError>;
}

impl<R: Runtime> GraphBackend for RuntimeGraph<R> {
    unsafe fn replay(&self) -> Result<(), ServerError> {
        unsafe { RuntimeGraph::replay(self) }
    }
}

trait ClientBackend: Send + Sync {
    fn clone_box(&self) -> Box<dyn ClientBackend>;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn name(&self) -> &'static str;
    fn service_id(&self) -> ServiceId;
    fn execution_stream(&self) -> StreamId;
    unsafe fn set_stream(&mut self, stream: StreamId);
    fn properties(&self) -> &DeviceProperties;
    fn properties_mut(&mut self) -> Option<&mut DeviceProperties>;
    fn properties_fingerprint(&self) -> u64;
    fn target_properties(&self) -> TargetProperties;
    fn can_read_tensor(&self, shape: &Shape, strides: &Strides) -> bool;
    fn utilization(&self) -> Result<DeviceUtilization, UtilizationUnavailable>;
    fn read(&self, descriptors: Vec<CopyDescriptor>) -> DynFut<Result<Vec<Bytes>, ServerError>>;
    fn create(&self, descriptors: Vec<MemoryLayoutDescriptor>, data: Vec<Bytes>)
        -> Result<Vec<MemoryLayout>, IoError>;
    fn create_from_slices(&self, descriptors: Vec<MemoryLayoutDescriptor>, data: Vec<Vec<u8>>)
        -> Result<Vec<MemoryLayout>, IoError>;
    fn empty(&self, descriptors: Vec<MemoryLayoutDescriptor>) -> Result<Vec<MemoryLayout>, IoError>;
    fn try_empty(&self, size: usize) -> Result<Handle, ServerError>;
    fn staging(&self, bytes: Vec<&mut Bytes>, file_only: bool);
    fn launch(&self, kernel: Box<dyn RudaKernel>, count: RudaCount, args: KernelArguments);
    unsafe fn launch_unchecked(&self, kernel: Box<dyn RudaKernel>, count: RudaCount, args: KernelArguments);
    fn flush(&self) -> Result<(), ServerError>;
    fn check(&self, handles: &[Handle]) -> Result<(), ServerError>;
    fn sync_buffers(&self, handles: &[Handle]) -> DynFut<Result<(), ServerError>>;
    fn sync(&self) -> DynFut<Result<(), ServerError>>;
    fn memory_usage(&self) -> Result<MemoryUsage, ServerError>;
    fn memory_cleanup(&self);
    fn memory_compact(&self) -> Result<(), ServerError>;
    unsafe fn allocation_mode(&self, mode: MemoryAllocationMode);
    fn allocation_window(&self, mode: MemoryAllocationMode, task: Box<dyn FnOnce() + Send + '_>);
    fn exclusive(&self, task: Task<'_>) -> Result<Value, ServerError>;
    fn profile(&self, task: Task<'_>, name: &str) -> Result<(Value, ProfileDuration), ProfileError>;
    fn profile_start(&self) -> Result<ProfileWindow, ProfileError>;
    fn profile_end(&self, window: ProfileWindow) -> Result<ProfileDuration, ProfileError>;
    fn profile_abandon(&self, window: ProfileWindow);
    fn measure_throughput(&self, key: ThroughputKey,
        probe: Box<dyn FnOnce() -> Result<ThroughputValue, ThroughputError> + Send + '_>)
        -> Result<ThroughputValue, ThroughputError>;
    fn enumerate_devices(&self, type_id: u16) -> Vec<DeviceId>;
    fn enumerate_all_devices(&self) -> Vec<DeviceId>;
    fn has_device_transport(&self) -> bool;
    fn ensure_init_collective(&mut self, devices: Vec<DeviceId>);
    fn sync_collective(&self);
    fn all_reduce(&mut self, src: Handle, dst: Handle, dtype: ElemType,
        devices: Vec<DeviceId>, op: ReduceOperation);
    fn to_client_tensor(&mut self, src: CopyDescriptor, dst: &dyn ClientBackend, dtype: ElemType)
        -> Handle;
    fn graph_prepare(&self) -> Result<(), ServerError>;
    fn graph_start_capture(&self) -> Result<(), ServerError>;
    fn graph_stop_capture(&self) -> Result<Graph, ServerError>;
    fn graph_abort_capture(&self);
}

fn typed_service<R: Runtime>(backend: &dyn ClientBackend) -> Result<ComputeClient<R>, ServerError> {
    let service = backend.service_id();
    if service.service != TypeId::of::<R::Server>() {
        return Err(ServerError::ServiceMismatch {
            client: alloc::format!("{service}"),
            requested: core::any::type_name::<R::Server>().into(),
            backtrace: BackTrace::capture(),
        });
    }
    let mut client = ComputeClient::<R>::load_service(service.device);
    unsafe { client.set_stream(backend.execution_stream()); }
    Ok(client)
}

impl<R: Runtime> ClientBackend for ComputeClient<R> {
    fn clone_box(&self) -> Box<dyn ClientBackend> { Box::new(self.clone()) }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn name(&self) -> &'static str { R::name(self) }
    fn service_id(&self) -> ServiceId { ComputeClient::service_id(self) }
    fn execution_stream(&self) -> StreamId { ComputeClient::execution_stream(self) }
    unsafe fn set_stream(&mut self, stream: StreamId) { unsafe { ComputeClient::set_stream(self, stream) } }
    fn properties(&self) -> &DeviceProperties { ComputeClient::properties(self) }
    fn properties_mut(&mut self) -> Option<&mut DeviceProperties> { ComputeClient::properties_mut(self) }
    fn properties_fingerprint(&self) -> u64 { ComputeClient::properties_fingerprint(self) }
    fn target_properties(&self) -> TargetProperties { R::target_properties() }
    fn can_read_tensor(&self, shape: &Shape, strides: &Strides) -> bool { R::can_read_tensor(shape, strides) }
    fn utilization(&self) -> Result<DeviceUtilization, UtilizationUnavailable> {
        R::utilization(&R::Device::from_id(self.device_id()))
    }
    fn read(&self, descriptors: Vec<CopyDescriptor>) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        self.do_read(descriptors)
    }
    fn create(&self, descriptors: Vec<MemoryLayoutDescriptor>, data: Vec<Bytes>)
        -> Result<Vec<MemoryLayout>, IoError> { self.do_create(descriptors, data) }
    fn create_from_slices(&self, descriptors: Vec<MemoryLayoutDescriptor>, data: Vec<Vec<u8>>)
        -> Result<Vec<MemoryLayout>, IoError> { self.do_create_from_slices(descriptors, data) }
    fn empty(&self, descriptors: Vec<MemoryLayoutDescriptor>) -> Result<Vec<MemoryLayout>, IoError> {
        self.do_empty(descriptors)
    }
    fn try_empty(&self, size: usize) -> Result<Handle, ServerError> { ComputeClient::try_empty(self, size) }
    fn staging(&self, bytes: Vec<&mut Bytes>, file_only: bool) {
        ComputeClient::staging(self, bytes.into_iter(), file_only)
    }
    fn launch(&self, kernel: Box<dyn RudaKernel>, count: RudaCount, args: KernelArguments) {
        ComputeClient::launch(self, Box::new(KernelTask::<R::Compiler, _>::new(kernel)), count, args)
    }
    unsafe fn launch_unchecked(&self, kernel: Box<dyn RudaKernel>, count: RudaCount, args: KernelArguments) {
        unsafe { ComputeClient::launch_unchecked(self,
            Box::new(KernelTask::<R::Compiler, _>::new(kernel)), count, args) }
    }
    fn flush(&self) -> Result<(), ServerError> { ComputeClient::flush(self) }
    fn check(&self, handles: &[Handle]) -> Result<(), ServerError> { ComputeClient::check(self, handles) }
    fn sync_buffers(&self, handles: &[Handle]) -> DynFut<Result<(), ServerError>> {
        ComputeClient::sync_buffers(self, handles)
    }
    fn sync(&self) -> DynFut<Result<(), ServerError>> { ComputeClient::sync(self) }
    fn memory_usage(&self) -> Result<MemoryUsage, ServerError> { ComputeClient::memory_usage(self) }
    fn memory_cleanup(&self) { ComputeClient::memory_cleanup(self) }
    fn memory_compact(&self) -> Result<(), ServerError> { ComputeClient::memory_compact(self) }
    unsafe fn allocation_mode(&self, mode: MemoryAllocationMode) {
        unsafe { ComputeClient::allocation_mode(self, mode) }
    }
    fn allocation_window(&self, mode: MemoryAllocationMode, task: Box<dyn FnOnce() + Send + '_>) {
        ComputeClient::allocation_window(self, mode, (), |()| task())
    }
    fn exclusive(&self, task: Task<'_>) -> Result<Value, ServerError> {
        ComputeClient::exclusive(self, task)
    }
    fn profile(&self, task: Task<'_>, name: &str) -> Result<(Value, ProfileDuration), ProfileError> {
        ComputeClient::profile(self, task, name)
    }
    fn profile_start(&self) -> Result<ProfileWindow, ProfileError> { ComputeClient::profile_start(self) }
    fn profile_end(&self, window: ProfileWindow) -> Result<ProfileDuration, ProfileError> {
        ComputeClient::profile_end(self, window)
    }
    fn profile_abandon(&self, window: ProfileWindow) { ComputeClient::profile_abandon(self, window) }
    fn measure_throughput(&self, key: ThroughputKey,
        probe: Box<dyn FnOnce() -> Result<ThroughputValue, ThroughputError> + Send + '_>)
        -> Result<ThroughputValue, ThroughputError> {
        ComputeClient::measure_throughput(self, key, probe)
    }
    fn enumerate_devices(&self, type_id: u16) -> Vec<DeviceId> { ComputeClient::enumerate_devices(self, type_id) }
    fn enumerate_all_devices(&self) -> Vec<DeviceId> { ComputeClient::enumerate_all_devices(self) }
    fn has_device_transport(&self) -> bool { R::Server::SERVER_COMM_ENABLED }
    fn ensure_init_collective(&mut self, devices: Vec<DeviceId>) {
        ComputeClient::ensure_init_collective(self, devices)
    }
    fn sync_collective(&self) { ComputeClient::sync_collective(self) }
    fn all_reduce(&mut self, src: Handle, dst: Handle, dtype: ElemType,
        devices: Vec<DeviceId>, op: ReduceOperation) {
        ComputeClient::all_reduce(self, src, dst, dtype, devices, op)
    }
    fn to_client_tensor(&mut self, src: CopyDescriptor, dst: &dyn ClientBackend, dtype: ElemType)
        -> Handle {
        let dst = typed_service::<R>(dst).expect("Device transport requires the same service type");
        ComputeClient::to_client_tensor(self, src, &dst, dtype)
    }
    fn graph_prepare(&self) -> Result<(), ServerError> { ComputeClient::graph_prepare(self) }
    fn graph_start_capture(&self) -> Result<(), ServerError> { ComputeClient::graph_start_capture(self) }
    fn graph_stop_capture(&self) -> Result<Graph, ServerError> {
        ComputeClient::graph_stop_capture(self).map(|graph| Graph { inner: Arc::new(graph) })
    }
    fn graph_abort_capture(&self) { ComputeClient::graph_abort_capture(self) }
}

impl ErasedClient {
    /// Register a configured server and erase its client type.
    pub fn init<R: Runtime, D: Device>(device: &D, server: R::Server) -> Self {
        ComputeClient::<R>::init(device, server).erase()
    }

    /// Register a configured server, retaining the original duplicate-registration error.
    pub fn try_init<R: Runtime, D: Device>(device: &D, server: R::Server)
        -> Result<Self, ruda_core::device_handle::ServiceCreationError> {
        ComputeClient::<R>::try_init(device, server).map(ComputeClient::erase)
    }

    /// Initialize a runtime and erase its client type.
    pub fn load<R: Runtime>(device: &R::Device) -> Self { R::client(device).erase() }

    /// Inspect a typed client only when it belongs to the requested runtime.
    pub fn as_typed<R: Runtime>(&self) -> Option<&ComputeClient<R>> {
        self.backend.as_any().downcast_ref()
    }

    /// Mutably access a matching typed client.
    pub fn as_typed_mut<R: Runtime>(&mut self) -> Option<&mut ComputeClient<R>> {
        self.backend.as_any_mut().downcast_mut()
    }

    /// Name used by this runtime's logs and caches.
    pub fn name(&self) -> &'static str { self.backend.name() }
    /// The service type and runtime-local device this client reaches.
    pub fn service_id(&self) -> ServiceId { self.backend.service_id() }
    /// Borrow this service through a compatible native runtime wrapper.
    /// Checks the concrete server type and retains the selected execution stream.
    pub fn typed_service<R: Runtime>(&self) -> Result<ComputeClient<R>, ServerError> {
        typed_service::<R>(self.backend.as_ref())
    }
    /// The runtime-local device ID, not a global catalogue ID.
    pub fn device_id(&self) -> DeviceId { self.service_id().device }
    /// Current logical stream, resolving thread-local selection on the caller.
    pub fn execution_stream(&self) -> StreamId { self.backend.execution_stream() }
    /// Whether both clients address the same service and logical stream.
    pub fn same_execution_queue(&self, other: &Self) -> bool {
        self.service_id() == other.service_id() && self.execution_stream() == other.execution_stream()
    }
    /// Clone with the currently resolved stream pinned.
    pub fn fixed_execution_queue(&self) -> Self {
        let mut client = self.clone();
        unsafe { client.set_stream(self.execution_stream()); }
        client
    }
    /// Select a stream explicitly.
    ///
    /// # Safety
    /// The caller must maintain ordering of all buffer accesses across streams.
    pub unsafe fn set_stream(&mut self, stream: StreamId) { unsafe { self.backend.set_stream(stream) } }
    /// Device capabilities and physical identity.
    pub fn properties(&self) -> &DeviceProperties { self.backend.properties() }
    /// Supported kernel features.
    pub fn features(&self) -> &Features { &self.properties().features }
    /// Mutable properties only when their shared storage is uniquely owned.
    pub fn properties_mut(&mut self) -> Option<&mut DeviceProperties> { self.backend.properties_mut() }
    /// Cached property checksum.
    pub fn properties_fingerprint(&self) -> u64 { self.backend.properties_fingerprint() }
    /// Compiler target layout properties.
    pub fn target_properties(&self) -> TargetProperties { self.backend.target_properties() }
    /// Whether the runtime can directly read this tensor layout.
    pub fn can_read_tensor(&self, shape: &Shape, strides: &Strides) -> bool {
        self.backend.can_read_tensor(shape, strides)
    }
    /// Query driver/platform utilization without synchronizing device work.
    pub fn utilization(&self) -> Result<DeviceUtilization, UtilizationUnavailable> {
        self.backend.utilization()
    }
    /// Read raw buffers asynchronously.
    pub fn read_async(&self, handles: Vec<Handle>) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        let descriptors = handles.into_iter().map(|handle| {
            let shape = [handle.size_in_used() as usize].into();
            handle.copy_descriptor(shape, [1].into(), 1)
        }).collect();
        self.backend.read(descriptors)
    }
    /// Read raw buffers, returning their actual read error.
    pub fn read(&self, handles: Vec<Handle>) -> Result<Vec<Bytes>, ServerError> {
        ruda_core::reader::read_sync(self.read_async(handles))
    }
    /// Read one raw buffer.
    pub fn read_one(&self, handle: Handle) -> Result<Bytes, ServerError> {
        Ok(self.read(vec![handle])?.remove(0))
    }
    /// Read one raw buffer, panicking on failure.
    pub fn read_one_unchecked(&self, handle: Handle) -> Bytes { self.read_one(handle).unwrap() }
    /// Read tensors using their explicit shape and stride descriptors.
    pub fn read_tensor_async(&self, descriptors: Vec<CopyDescriptor>) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        self.backend.read(descriptors)
    }
    /// Read tensor descriptors synchronously.
    pub fn read_tensor(&self, descriptors: Vec<CopyDescriptor>) -> Result<Vec<Bytes>, ServerError> {
        ruda_core::reader::read_sync(self.read_tensor_async(descriptors))
    }
    /// Read one tensor asynchronously.
    pub fn read_one_tensor_async(&self, descriptor: CopyDescriptor) -> DynFut<Result<Bytes, ServerError>> {
        let read = self.read_tensor_async(vec![descriptor]);
        Box::pin(async move { Ok(read.await?.remove(0)) })
    }
    /// Read one tensor, panicking on failure.
    pub fn read_one_unchecked_tensor(&self, descriptor: CopyDescriptor) -> Bytes {
        self.read_tensor(vec![descriptor]).unwrap().remove(0)
    }
    /// Obtain a native resource after checking the requested service type.
    pub fn get_resource<R: Runtime>(&self, handle: Handle)
        -> Result<ManagedResource<<<R::Server as ComputeServer>::Storage as ComputeStorage>::Resource>, ServerError> {
        typed_service::<R>(self.backend.as_ref())?.get_resource(handle)
    }
    /// Create a contiguous raw buffer from owned bytes.
    pub fn create(&self, data: Bytes) -> Handle {
        let descriptor = MemoryLayoutDescriptor::new(MemoryLayoutStrategy::Contiguous, [data.len()].into(), 1);
        self.backend.create(vec![descriptor], vec![data]).unwrap().remove(0).memory
    }
    /// Create a contiguous raw buffer from borrowed bytes.
    pub fn create_from_slice(&self, data: &[u8]) -> Handle {
        let descriptor = MemoryLayoutDescriptor::new(MemoryLayoutStrategy::Contiguous, [data.len()].into(), 1);
        self.backend.create_from_slices(vec![descriptor], vec![data.to_vec()]).unwrap().remove(0).memory
    }
    /// Create a tensor with the runtime's optimized allocation layout.
    pub fn create_tensor(&self, data: Bytes, shape: Shape, elem_size: usize) -> MemoryLayout {
        self.create_tensors(vec![(MemoryLayoutDescriptor::new(MemoryLayoutStrategy::Optimized, shape, elem_size), data)]).remove(0)
    }
    /// Create a tensor from borrowed bytes.
    pub fn create_tensor_from_slice(&self, data: &[u8], shape: Shape, elem_size: usize) -> MemoryLayout {
        self.create_tensors_from_slices(vec![(MemoryLayoutDescriptor::new(MemoryLayoutStrategy::Optimized, shape, elem_size), data)]).remove(0)
    }
    /// Allocate and upload a group of tensors using the original layout policy.
    pub fn create_tensors(&self, descriptors: Vec<(MemoryLayoutDescriptor, Bytes)>) -> Vec<MemoryLayout> {
        let (descriptors, bytes) = descriptors.into_iter().unzip();
        self.backend.create(descriptors, bytes).unwrap()
    }
    /// Allocate and upload tensors from borrowed byte slices.
    pub fn create_tensors_from_slices(&self, descriptors: Vec<(MemoryLayoutDescriptor, &[u8])>) -> Vec<MemoryLayout> {
        let (descriptors, bytes) = descriptors.into_iter().map(|(desc, data)| (desc, data.to_vec())).unzip();
        self.backend.create_from_slices(descriptors, bytes).unwrap()
    }
    /// Reserve a raw contiguous buffer.
    pub fn empty(&self, size: usize) -> Handle {
        self.empty_tensors(vec![MemoryLayoutDescriptor::new(MemoryLayoutStrategy::Contiguous, [size].into(), 1)]).remove(0).memory
    }
    /// Reserve a raw buffer and return allocation errors.
    pub fn try_empty(&self, size: usize) -> Result<Handle, ServerError> { self.backend.try_empty(size) }
    /// Reserve an optimized tensor layout.
    pub fn empty_tensor(&self, shape: Shape, elem_size: usize) -> MemoryLayout {
        self.empty_tensors(vec![MemoryLayoutDescriptor::new(MemoryLayoutStrategy::Optimized, shape, elem_size)]).remove(0)
    }
    /// Reserve a group of tensor layouts.
    pub fn empty_tensors(&self, descriptors: Vec<MemoryLayoutDescriptor>) -> Vec<MemoryLayout> {
        self.backend.empty(descriptors).unwrap()
    }
    /// Convert eligible host buffers to the runtime's staging allocation.
    pub fn staging<'a>(&self, bytes: impl Iterator<Item = &'a mut Bytes>, file_only: bool) {
        self.backend.staging(bytes.collect(), file_only)
    }
    /// Compile and submit a runtime-independent kernel with bounds checks.
    pub fn launch(&self, kernel: Box<dyn RudaKernel>, count: RudaCount, args: KernelArguments) {
        self.backend.launch(kernel, count, args)
    }
    /// Compile and submit without implicit bounds checks, subject to configured enforcement.
    ///
    /// # Safety
    /// The kernel must not access out-of-bounds memory or contain nonterminating loops.
    pub unsafe fn launch_unchecked(&self, kernel: Box<dyn RudaKernel>, count: RudaCount, args: KernelArguments) {
        unsafe { self.backend.launch_unchecked(kernel, count, args) }
    }
    /// Submit outstanding work without waiting for device completion.
    pub fn flush(&self) -> Result<(), ServerError> { self.backend.flush() }
    /// Inspect buffer failures without a completion fence.
    pub fn check<'a>(&self, handles: impl IntoIterator<Item = &'a Handle>) -> Result<(), ServerError> {
        self.backend.check(&handles.into_iter().cloned().collect::<Vec<_>>())
    }
    /// Wait for selected buffers and inspect their failures.
    pub fn sync_buffers<'a>(&self, handles: impl IntoIterator<Item = &'a Handle>) -> DynFut<Result<(), ServerError>> {
        self.backend.sync_buffers(&handles.into_iter().cloned().collect::<Vec<_>>())
    }
    /// Return the device-completion future.
    pub fn sync(&self) -> DynFut<Result<(), ServerError>> { self.backend.sync() }
    /// Allocator accounting for this runtime.
    pub fn memory_usage(&self) -> Result<MemoryUsage, ServerError> { self.backend.memory_usage() }
    /// Release allocations the runtime can reclaim.
    pub fn memory_cleanup(&self) { self.backend.memory_cleanup() }
    /// Compact supported adaptive allocations.
    pub fn memory_compact(&self) -> Result<(), ServerError> { self.backend.memory_compact() }
    /// Select an allocation mode.
    ///
    /// # Safety
    /// The caller must prevent concurrent conflicting changes and restore the intended mode.
    pub unsafe fn allocation_mode(&self, mode: MemoryAllocationMode) {
        unsafe { self.backend.allocation_mode(mode) }
    }
    /// Perform a closure in persistent allocation mode.
    pub fn memory_persistent_allocation<I: Send, O: Send>(
        &self, input: I, task: impl FnOnce(I) -> O + Send,
    ) -> Result<O, ServerError> {
        Ok(self.allocation_window(MemoryAllocationMode::Persistent, input, task))
    }
    /// Perform a closure using dedicated allocations.
    pub fn memory_dedicated_allocation<I: Send, O: Send>(
        &self, input: I, task: impl FnOnce(I) -> O + Send,
    ) -> O { self.allocation_window(MemoryAllocationMode::Dedicated, input, task) }
    fn allocation_window<I: Send, O: Send>(
        &self, mode: MemoryAllocationMode, input: I, task: impl FnOnce(I) -> O + Send,
    ) -> O {
        let mut output = None;
        self.backend.allocation_window(mode, Box::new(|| output = Some(task(input))));
        output.expect("Allocation closure was not invoked")
    }
    /// Execute under the original service's exclusive access mechanism.
    pub fn exclusive<O: Send + 'static>(&self, task: impl FnOnce() -> O + Send) -> Result<O, ServerError> {
        let value = self.backend.exclusive(Box::new(|| Box::new(task())))?;
        Ok(*value.downcast::<O>().expect("Exclusive result type mismatch"))
    }
    /// Profile a closure using the original service's profiler.
    pub fn profile<O: Send + 'static>(&self, task: impl FnOnce() -> O + Send, name: &str)
        -> Result<(O, ProfileDuration), ProfileError> {
        let (value, duration) = self.backend.profile(Box::new(|| Box::new(task())), name)?;
        Ok((*value.downcast::<O>().expect("Profile result type mismatch"), duration))
    }
    /// Open a profiling window.
    pub fn profile_start(&self) -> Result<ProfileWindow, ProfileError> { self.backend.profile_start() }
    /// Close a window on its original stream.
    pub fn profile_end(&self, window: ProfileWindow) -> Result<ProfileDuration, ProfileError> {
        self.backend.profile_end(window)
    }
    /// Abandon a profiling window.
    pub fn profile_abandon(&self, window: ProfileWindow) { self.backend.profile_abandon(window) }
    /// Query or measure a throughput ceiling using this service's existing cache.
    pub fn measure_throughput(&self, key: ThroughputKey,
        probe: impl FnOnce() -> Result<ThroughputValue, ThroughputError> + Send)
        -> Result<ThroughputValue, ThroughputError> {
        self.backend.measure_throughput(key, Box::new(probe))
    }
    /// Enumerate runtime-local IDs of one device type.
    pub fn enumerate_devices(&self, type_id: u16) -> Vec<DeviceId> { self.backend.enumerate_devices(type_id) }
    /// Enumerate all devices of the selected runtime/API.
    pub fn enumerate_all_devices(&self) -> Vec<DeviceId> { self.backend.enumerate_all_devices() }
    /// Number of devices of one runtime-local type.
    pub fn device_count(&self, type_id: u16) -> usize { self.enumerate_devices(type_id).len() }
    /// Number of devices visible to this runtime/API.
    pub fn device_count_total(&self) -> usize { self.enumerate_all_devices().len() }
    /// Whether this runtime supplies native inter-device transport.
    pub fn has_device_transport(&self) -> bool { self.backend.has_device_transport() }
    /// Initialize native collective communication.
    pub fn ensure_init_collective(&mut self, devices: Vec<DeviceId>) {
        assert!(self.has_device_transport(), "This runtime has no native device transport");
        self.backend.ensure_init_collective(devices)
    }
    /// Wait for collective submission, doing nothing without native transport.
    pub fn sync_collective(&self) {
        if self.has_device_transport() { self.backend.sync_collective(); }
    }
    /// Reduce local input/output buffers using native collective transport.
    pub fn all_reduce(&mut self, src: Handle, dst: Handle, dtype: ElemType,
        devices: Vec<DeviceId>, op: ReduceOperation) {
        assert!(self.has_device_transport(), "This runtime has no native device transport");
        self.backend.all_reduce(src, dst, dtype, devices, op)
    }
    /// Transfer raw bytes natively within a compatible runtime, otherwise through host memory.
    pub fn to_client(&mut self, src: Handle, dst: &Self, dtype: ElemType) -> Handle {
        let shape = [src.size_in_used() as usize].into();
        let descriptor = src.copy_descriptor(shape, [1].into(), 1);
        if self.has_device_transport() && self.service_id().service == dst.service_id().service {
            self.to_client_tensor(descriptor, dst, dtype)
        } else {
            let allocation = MemoryLayoutDescriptor::new(MemoryLayoutStrategy::Contiguous,
                descriptor.shape.clone(), descriptor.elem_size);
            let data = self.read_tensor(vec![descriptor]).unwrap().remove(0);
            dst.create_tensors(vec![(allocation, data)]).remove(0).memory
        }
    }
    /// Transfer a tensor through the runtime's native device transport.
    pub fn to_client_tensor(&mut self, src: CopyDescriptor, dst: &Self, dtype: ElemType) -> Handle {
        assert!(self.has_device_transport() && dst.has_device_transport(),
            "Tensor device transport requires native transport on both clients");
        assert_eq!(self.service_id().service, dst.service_id().service,
            "Tensor device transport requires the same service type");
        self.backend.to_client_tensor(src, dst.backend.as_ref(), dtype)
    }
    /// Prepare a graph-capture warmup.
    pub fn graph_prepare(&self) -> Result<(), ServerError> { self.backend.graph_prepare() }
    /// Begin recording warmed work.
    pub fn graph_start_capture(&self) -> Result<(), ServerError> { self.backend.graph_start_capture() }
    /// Seal a graph while retaining its capture stream and service.
    pub fn graph_stop_capture(&self) -> Result<Graph, ServerError> { self.backend.graph_stop_capture() }
    /// Abort the current recording.
    pub fn graph_abort_capture(&self) { self.backend.graph_abort_capture() }
    /// Vector sizes useful for memory operations on elements of `size` bytes.
    pub fn io_optimized_vector_sizes(&self, size: usize) -> impl Iterator<Item = VectorSize> + Clone {
        self.properties().io_optimized_vector_sizes(size)
    }
}
