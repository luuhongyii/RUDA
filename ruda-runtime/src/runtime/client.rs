use crate::runtime::{
    config::{TypeNameFormatLevel, type_name_format},
    kernel::KernelMetadata,
    logging::ProfileLevel,
    memory_management::{MemoryAllocationMode, MemoryUsage},
    backend::Runtime,
    server::{
        Binding, Collective, CommunicationId, ComputeServer, CopyDescriptor, RudaCount, ExecutionMode, Handle, IoError,
        KernelArguments, MemoryLayout, MemoryLayoutDescriptor, MemoryLayoutPolicy,
        MemoryLayoutStrategy, ProfileError, ReduceOperation, ServerCommunication, ServerError,
        ServerUtilities,
    },
    storage::{ComputeStorage, ManagedResource},
};
use alloc::{format, sync::Arc, vec, vec::Vec};
use ruda_core::{
    backtrace::BackTrace,
    bytes::{AllocationProperty, Bytes},
    device::{Device, DeviceId},
    device_handle::DeviceHandle,
    future::DynFut,
    profile::ProfileDuration,
};
use ruda_core::ir::{DeviceProperties, ElemType, VectorSize, features::Features};
use ruda_core::tensor::Shape;

mod erased;
pub use erased::{Client, ErasedClient, Graph};

#[allow(unused)]
use ruda_core::profile::TimingMethod;
use ruda_core::stream_id::StreamId;

/// An open profiling window and the exact logical stream that owns it.
#[derive(Debug, Clone, Copy)]
pub struct ProfileWindow {
    /// Stream on which the start mark was recorded.
    pub stream_id: StreamId,
    /// Server-issued window token.
    pub token: crate::runtime::server::ProfilingToken,
}

struct PendingProfile<S: ComputeServer> { device: DeviceHandle<S>, window: Option<ProfileWindow> }
impl<S: ComputeServer> Drop for PendingProfile<S> {
    fn drop(&mut self) {
        if let Some(window) = self.window.take() {
            self.device.submit(move |server| server.abandon_profile(window.stream_id, window.token));
        }
    }
}

struct AllocationWindow<S: ComputeServer> { device: DeviceHandle<S>, stream_id: StreamId }
impl<S: ComputeServer> Drop for AllocationWindow<S> {
    fn drop(&mut self) {
        let stream_id = self.stream_id;
        self.device.submit(move |server| server.allocation_mode(MemoryAllocationMode::Auto, stream_id));
    }
}

/// The `ComputeClient` is the entry point to require tasks from the `ComputeServer`.
/// It should be obtained for a specific device via the Compute struct.
pub struct ComputeClient<R: Runtime> {
    device: DeviceHandle<R::Server>,
    utilities: Arc<ServerUtilities<R::Server>>,
    stream_id: Option<StreamId>,
}

impl<R: Runtime> Clone for ComputeClient<R> {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            utilities: self.utilities.clone(),
            stream_id: self.stream_id,
        }
    }
}

impl<R: Runtime> ComputeClient<R> {
    /// Erase the runtime type while retaining this device service and stream selection.
    pub fn erase(self) -> ErasedClient { ErasedClient::from(self) }

    /// The concrete service and runtime-local device reached by this client.
    pub fn service_id(&self) -> ruda_core::device::ServiceId { self.device.service_id() }

    fn local(&self, binding: &Binding) -> Result<(), ServerError> {
        let client = self.service_id();
        if binding.service == client {
            return Ok(());
        }
        Err(ServerError::ForeignHandle {
            handle: format!("{}", binding.service),
            client: format!("{client}"),
            backtrace: BackTrace::capture(),
        })
    }

    #[track_caller]
    fn expect_local(&self, binding: &Binding) {
        if let Err(error) = self.local(binding) {
            panic!("{error}");
        }
    }

    fn bindings<'a>(
        &self,
        handles: impl IntoIterator<Item = &'a Handle>,
    ) -> Result<Vec<Binding>, ServerError> {
        handles.into_iter().map(|handle| {
            let binding = handle.clone().binding();
            self.local(&binding)?;
            Ok(binding)
        }).collect()
    }

    /// Get the info of the current backend.
    pub fn info(&self) -> &<R::Server as ComputeServer>::Info {
        &self.utilities.info
    }

    pub fn runtime_name(&self) -> Option<&'static str> {
        self.utilities.runtime_name
    }

    /// Physical runtime device ordinal/type used by tuning identity probes.
    pub fn device_id(&self) -> DeviceId { self.device.device_id() }

    /// Cached capability/hardware checksum. Does not probe the driver on every operator call.
    pub fn properties_fingerprint(&self) -> u64 { self.utilities.properties_hash }

    pub fn measure_throughput(
        &self,
        key: super::throughput::ThroughputKey,
        probe: impl FnOnce() -> Result<
            super::throughput::ThroughputValue,
            super::throughput::ThroughputError,
        > + Send,
    ) -> Result<super::throughput::ThroughputValue, super::throughput::ThroughputError> {
        use super::throughput::{ThroughputBenchmarker, ThroughputCache, ThroughputError};
        let cache = ThroughputCache::get_for_device(R::name(self), self.properties());
        let mut measurements = ThroughputBenchmarker::new(cache);
        if let Some(value) = measurements.cached(key) {
            return Ok(value);
        }
        self.exclusive(move || measurements.measure(key, probe))
            .unwrap_or(Err(ThroughputError::Launch))
    }

    /// Create a new client with a new server.
    ///
    /// Registers this device/server pair. Panics if that server type is already
    /// registered for the device; ordinary callers should use Runtime::client.
    pub fn init<D: Device>(device: &D, server: R::Server) -> Self {
        Self::try_init(device, server)
            .expect("Can't create a new client on an already registered server")
    }

    pub fn try_init<D: Device>(
        device: &D,
        server: R::Server,
    ) -> Result<Self, ruda_core::device_handle::ServiceCreationError> {
        let utilities = server.utilities();
        let context = DeviceHandle::<R::Server>::insert(device.to_id(), server)?;

        Ok(Self {
            device: context,
            utilities,
            stream_id: None,
        })
    }

    /// Load the client for the given device.
    /// The runtime must already have initialized a compatible server for that
    /// device. This does not construct a server or select a different physical GPU.
    pub fn load<D: Device>(device: &D) -> Self {
        Self::load_service(device.to_id())
    }

    fn load_service(device_id: DeviceId) -> Self {
        let context = DeviceHandle::<R::Server>::new(device_id);

        // This is safe because we now know the return type of [`DeviceHandle::utilities()`].
        let utilities = context
            .utilities()
            .downcast::<ServerUtilities<R::Server>>()
            .expect("Can downcast to `ServerUtilities`");

        Self {
            device: context,
            utilities,
            stream_id: None,
        }
    }

    fn stream_id(&self) -> StreamId {
        match self.stream_id {
            Some(val) => val,
            None => StreamId::current(),
        }
    }

    /// Resolve the logical execution stream on the calling thread.
    /// A reusable execution plan must also hold `fixed_execution_queue()`;
    /// an implicit thread-local stream must not change between plan operations.
    pub fn execution_stream(&self) -> StreamId { self.stream_id() }

    /// Whether both clients currently submit to the same server and stream.
    /// Implicit streams are resolved on the calling thread at the time of this check.
    pub fn same_execution_queue(&self, other: &Self) -> bool {
        self.device.device_id() == other.device.device_id() && self.stream_id() == other.stream_id()
    }

    /// Clone this client with its currently resolved execution queue fixed.
    /// Useful for reusable device plans whose scratch must not silently move
    /// between thread-local streams. Does not create a stream or synchronize.
    pub fn fixed_execution_queue(&self) -> Self {
        let mut client = self.clone();
        client.stream_id = Some(self.stream_id());
        client
    }

    /// Set the stream in which the current client is operating on.
    ///
    /// # Safety
    ///
    /// This is highly unsafe and should probably only be used by the Ruda/Ruda projects for now.
    pub unsafe fn set_stream(&mut self, stream_id: StreamId) {
        self.stream_id = Some(stream_id);
    }

    fn do_read(&self, descriptors: Vec<CopyDescriptor>) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        if let Some(error) = descriptors.iter().find_map(|descriptor| self.local(&descriptor.handle).err()) {
            return alloc::boxed::Box::pin(core::future::ready(Err(error)));
        }
        let stream_id = self.stream_id();
        let bindings: Vec<_> = descriptors.iter().map(|descriptor| descriptor.handle.clone()).collect();
        self.device
            .submit_blocking(move |server| -> DynFut<Result<Vec<Bytes>, ServerError>> {
                if let Err(error) = server.check(bindings, stream_id) {
                    return alloc::boxed::Box::pin(async move { Err(error) });
                }
                server.read(descriptors, stream_id)
            })
            .unwrap()
    }

    /// Given bindings, returns owned resources as bytes.
    pub fn read_async(
        &self,
        handles: Vec<Handle>,
    ) -> impl Future<Output = Result<Vec<Bytes>, ServerError>> + Send {
        let descriptors = handles
            .into_iter()
            .map(|handle| {
                let shape = [handle.size_in_used() as usize].into();
                CopyDescriptor::new(handle.binding(), shape, [1].into(), 1)
            })
            .collect();

        self.do_read(descriptors)
    }

    /// Given bindings, returns owned resources as bytes.
    ///
    /// # Remarks
    ///
    /// Panics if the read operation fails.
    pub fn read(&self, handles: Vec<Handle>) -> Vec<Bytes> {
        ruda_core::reader::read_sync(self.read_async(handles)).expect("TODO")
    }

    /// Given a binding, returns owned resource as bytes.
    /// Blocks for raw byte readback and returns device/read errors. Tensor
    /// shape/stride interpretation belongs to the tensor-descriptor interfaces.
    pub fn read_one(&self, handle: Handle) -> Result<Bytes, ServerError> {
        Ok(ruda_core::reader::read_sync(self.read_async(vec![handle]))?.remove(0))
    }

    /// Given a binding, returns owned resource as bytes.
    ///
    /// # Remarks
    ///
    /// Panics if the read operation fails. Useful for tests.
    pub fn read_one_unchecked(&self, handle: Handle) -> Bytes {
        ruda_core::reader::read_sync(self.read_async(vec![handle]))
            .unwrap()
            .remove(0)
    }

    /// Given bindings, returns owned resources as bytes.
    pub fn read_tensor_async(
        &self,
        descriptors: Vec<CopyDescriptor>,
    ) -> impl Future<Output = Result<Vec<Bytes>, ServerError>> + Send {
        self.do_read(descriptors)
    }

    /// Given bindings, returns owned resources as bytes.
    ///
    /// # Remarks
    ///
    /// Panics if the read operation fails.
    ///
    /// The tensor must be in the same layout as created by the runtime, or more strict.
    /// Contiguous tensors are always fine, strided tensors are only ok if the stride is similar to
    /// the one created by the runtime (i.e. padded on only the last dimension).
    /// Check layout compatibility with Runtime::can_read_tensor; make an
    /// unsupported layout contiguous before readback.
    ///
    /// Also see [`ComputeClient::create_tensor`].
    pub fn read_tensor(&self, descriptors: Vec<CopyDescriptor>) -> Vec<Bytes> {
        ruda_core::reader::read_sync(self.read_tensor_async(descriptors)).expect("TODO")
    }

    /// Given a binding, returns owned resource as bytes.
    /// See [`ComputeClient::read_tensor`]
    pub fn read_one_tensor_async(
        &self,
        descriptor: CopyDescriptor,
    ) -> impl Future<Output = Result<Bytes, ServerError>> + Send {
        let fut = self.read_tensor_async(vec![descriptor]);

        async { Ok(fut.await?.remove(0)) }
    }

    /// Given a binding, returns owned resource as bytes.
    ///
    /// # Remarks
    ///
    /// Panics if the read operation fails.
    /// See [`ComputeClient::read_tensor`]
    pub fn read_one_unchecked_tensor(&self, descriptor: CopyDescriptor) -> Bytes {
        self.read_tensor(vec![descriptor]).remove(0)
    }

    /// Given a resource handle, returns the storage resource.
    pub fn get_resource(
        &self,
        handle: Handle,
    ) -> Result<
        ManagedResource<<<R::Server as ComputeServer>::Storage as ComputeStorage>::Resource>,
        ServerError,
    > {
        let stream_id = self.stream_id();
        let binding = handle.binding();
        self.local(&binding)?;

        self.device
            .submit_blocking(move |state| {
                if let Some(error) = binding.failure() { return Err(error); }
                state.get_resource(binding, stream_id)
            })
            .map_err(|error| ServerError::Generic {
                reason: format!("Resource lookup failed: {error:?}"),
                backtrace: BackTrace::capture(),
            })?
    }

    fn do_create_from_slices(
        &self,
        descriptors: Vec<MemoryLayoutDescriptor>,
        slices: Vec<Vec<u8>>,
    ) -> Result<Vec<MemoryLayout>, IoError> {
        let stream_id = self.stream_id();
        let (handle_base, layouts) = self.utilities.layout_policy.apply(self.service_id(), stream_id, &descriptors);

        let descriptors = descriptors
            .into_iter()
            .zip(layouts.iter())
            .zip(slices)
            .map(|((desc, alloc), data)| {
                (
                    CopyDescriptor::new(
                        alloc.memory.clone().binding(),
                        desc.shape,
                        alloc.strides.clone(),
                        desc.elem_size,
                    ),
                    Bytes::from_bytes_vec(data),
                )
            })
            .collect::<Vec<_>>();

        let (size, memory) = (handle_base.size(), handle_base.memory);
        self.device.submit(move |server| {
            server.initialize_memory(memory, size, stream_id);
            server.write(descriptors, stream_id);
        });

        Ok(layouts)
    }

    fn do_create(
        &self,
        descriptors: Vec<MemoryLayoutDescriptor>,
        mut data: Vec<Bytes>,
    ) -> Result<Vec<MemoryLayout>, IoError> {
        self.staging(data.iter_mut(), true);

        let stream_id = self.stream_id();
        let (handle_base, layouts) = self.utilities.layout_policy.apply(self.service_id(), stream_id, &descriptors);

        let descriptors = descriptors
            .into_iter()
            .zip(layouts.iter())
            .zip(data)
            .map(|((desc, layout), data)| {
                (
                    CopyDescriptor::new(
                        layout.memory.clone().binding(),
                        desc.shape,
                        layout.strides.clone(),
                        desc.elem_size,
                    ),
                    data,
                )
            })
            .collect::<Vec<_>>();

        let (size, memory) = (handle_base.size(), handle_base.memory);
        self.device.submit(move |server| {
            server.initialize_memory(memory, size, stream_id);
            server.write(descriptors, stream_id);
        });

        Ok(layouts)
    }

    /// Returns a resource handle containing the given data.
    ///
    /// # Notes
    ///
    /// Prefer using the more efficient [`Self::create`] function.
    pub fn create_from_slice(&self, slice: &[u8]) -> Handle {
        let shape: Shape = [slice.len()].into();

        self.do_create_from_slices(
            vec![MemoryLayoutDescriptor::new(
                MemoryLayoutStrategy::Contiguous,
                shape,
                1,
            )],
            vec![slice.to_vec()],
        )
        .unwrap()
        .remove(0)
        .memory
    }

    /// todo: docs
    pub fn exclusive<'a, Re: Send + 'static, F: FnOnce() -> Re + Send + 'a>(
        &'a self,
        task: F,
    ) -> Result<Re, ServerError> {
        // We then launch the task.
        self.device
            .exclusive(task)
            .map_err(|err| ServerError::Generic {
                reason: format!("Communication channel with the server is down: {err:?}"),
                backtrace: BackTrace::capture(),
            })
    }

    /// dodo: Docs
    pub fn memory_persistent_allocation<
        'a,
        Re: Send,
        Input: Send,
        F: FnOnce(Input) -> Re + Send + 'a,
    >(
        &'a self,
        input: Input,
        task: F,
    ) -> Result<Re, ServerError> {
        Ok(self.allocation_window(MemoryAllocationMode::Persistent, input, task))
    }

    /// Allocate each temporary buffer outside the reusable pools, restoring the enclosing mode.
    pub fn memory_dedicated_allocation<'a, Re: Send, Input: Send, F>(
        &'a self,
        input: Input,
        task: F,
    ) -> Re
    where
        F: FnOnce(Input) -> Re + Send + 'a,
    {
        self.allocation_window(MemoryAllocationMode::Dedicated, input, task)
    }

    fn allocation_window<'a, Re: Send, Input: Send, F>(
        &'a self,
        mode: MemoryAllocationMode,
        input: Input,
        task: F,
    ) -> Re
    where
        F: FnOnce(Input) -> Re + Send + 'a,
    {
        let stream_id = self.stream_id();
        self.device.submit(move |server| server.allocation_mode(mode, stream_id));
        let _window = AllocationWindow { device: self.device.clone(), stream_id };
        task(input)
    }

    /// Returns a resource handle containing the given [Bytes].
    pub fn create(&self, data: Bytes) -> Handle {
        let shape = [data.len()].into();

        self.do_create(
            vec![MemoryLayoutDescriptor::new(
                MemoryLayoutStrategy::Contiguous,
                shape,
                1,
            )],
            vec![data],
        )
        .unwrap()
        .remove(0)
        .memory
    }

    /// Given a resource and shape, stores it and returns the tensor handle and strides.
    /// This may or may not return contiguous strides. The layout is up to the runtime, and care
    /// should be taken when indexing.
    ///
    /// Currently the tensor may either be contiguous (most runtimes), or "pitched", to use the CUDA
    /// terminology. This means the last (contiguous) dimension is padded to fit a certain alignment,
    /// and the strides are adjusted accordingly. This can make memory accesses significantly faster
    /// since all rows are aligned to at least 16 bytes (the maximum load width), meaning the GPU
    /// can load as much data as possible in a single instruction. It may be aligned even more to
    /// also take cache lines into account.
    ///
    /// However, the stride must be taken into account when indexing and reading the tensor
    /// (also see [`ComputeClient::read_tensor`]).
    ///
    /// # Notes
    ///
    /// Prefer using [`Self::create_tensor`] for better performance.
    pub fn create_tensor_from_slice(
        &self,
        slice: &[u8],
        shape: Shape,
        elem_size: usize,
    ) -> MemoryLayout {
        self.do_create_from_slices(
            vec![MemoryLayoutDescriptor::new(
                MemoryLayoutStrategy::Optimized,
                shape,
                elem_size,
            )],
            vec![slice.to_vec()],
        )
        .unwrap()
        .remove(0)
    }

    /// Given a resource and shape, stores it and returns the tensor handle and strides.
    /// This may or may not return contiguous strides. The layout is up to the runtime, and care
    /// should be taken when indexing.
    ///
    /// Currently the tensor may either be contiguous (most runtimes), or "pitched", to use the CUDA
    /// terminology. This means the last (contiguous) dimension is padded to fit a certain alignment,
    /// and the strides are adjusted accordingly. This can make memory accesses significantly faster
    /// since all rows are aligned to at least 16 bytes (the maximum load width), meaning the GPU
    /// can load as much data as possible in a single instruction. It may be aligned even more to
    /// also take cache lines into account.
    ///
    /// However, the stride must be taken into account when indexing and reading the tensor
    /// (also see [`ComputeClient::read_tensor`]).
    pub fn create_tensor(&self, bytes: Bytes, shape: Shape, elem_size: usize) -> MemoryLayout {
        self.do_create(
            vec![MemoryLayoutDescriptor::new(
                MemoryLayoutStrategy::Optimized,
                shape,
                elem_size,
            )],
            vec![bytes],
        )
        .unwrap()
        .remove(0)
    }

    /// Reserves all `shapes` in a single storage buffer, copies the corresponding `data` into each
    /// handle, and returns the handles for them.
    /// See [`ComputeClient::create_tensor`]
    ///
    /// # Notes
    ///
    /// Prefer using [`Self::create_tensors`] for better performance.
    pub fn create_tensors_from_slices(
        &self,
        descriptors: Vec<(MemoryLayoutDescriptor, &[u8])>,
    ) -> Vec<MemoryLayout> {
        let mut data = Vec::with_capacity(descriptors.len());
        let mut descriptors_ = Vec::with_capacity(descriptors.len());
        for (a, b) in descriptors {
            data.push(b.to_vec());
            descriptors_.push(a);
        }

        self.do_create_from_slices(descriptors_, data).unwrap()
    }

    /// Reserves all `shapes` in a single storage buffer, copies the corresponding `data` into each
    /// handle, and returns the handles for them.
    /// See [`ComputeClient::create_tensor`]
    pub fn create_tensors(
        &self,
        descriptors: Vec<(MemoryLayoutDescriptor, Bytes)>,
    ) -> Vec<MemoryLayout> {
        let (descriptors, data) = descriptors.into_iter().unzip();

        self.do_create(descriptors, data).unwrap()
    }

    fn do_empty(
        &self,
        descriptors: Vec<MemoryLayoutDescriptor>,
    ) -> Result<Vec<MemoryLayout>, IoError> {
        let stream_id = self.stream_id();
        let (handle_base, layouts) = self.utilities.layout_policy.apply(self.service_id(), stream_id, &descriptors);

        let (size, memory) = (handle_base.size(), handle_base.memory);
        self.device.submit(move |server| {
            server.initialize_memory(memory, size, stream_id);
        });

        Ok(layouts)
    }

    /// Reserves `size` bytes in the storage, and returns a handle over them.
    pub fn empty(&self, size: usize) -> Handle {
        let shape: Shape = [size].into();
        let descriptor = MemoryLayoutDescriptor::new(MemoryLayoutStrategy::Contiguous, shape, 1);
        self.do_empty(vec![descriptor]).unwrap().remove(0).memory
    }

    /// Reserve a buffer and wait for its allocation result, without copying its bytes.
    pub fn try_empty(&self, size: usize) -> Result<Handle, ServerError> {
        self.try_empty_with_resource(size).map(|(handle, _)| handle)
    }

    /// Reserve a buffer and resolve its resource in one device-service command.
    /// The returned resource retains the same storage lease as [`Self::get_resource`].
    pub fn try_empty_with_resource(
        &self,
        size: usize,
    ) -> Result<
        (Handle, ManagedResource<<<R::Server as ComputeServer>::Storage as ComputeStorage>::Resource>),
        ServerError,
    > {
        let shape: Shape = [size].into();
        let descriptor = MemoryLayoutDescriptor::new(MemoryLayoutStrategy::Contiguous, shape, 1);
        let stream_id = self.stream_id();
        let (base, mut layouts) = self.utilities.layout_policy.apply(self.service_id(), stream_id, &[descriptor]);
        let handle = layouts.remove(0).memory;
        let binding = handle.clone().binding();
        self.local(&binding)?;
        let (size, memory) = (base.size(), base.memory);
        let resource = self.device.submit_blocking(move |server| {
            server.initialize_memory(memory, size, stream_id);
            if let Some(error) = binding.failure() { return Err(error); }
            server.get_resource(binding, stream_id)
        }).map_err(|error| ServerError::Generic {
            reason: format!("Allocation resource lookup failed: {error:?}"),
            backtrace: BackTrace::capture(),
        })??;
        Ok((handle, resource))
    }

    /// Reserves `shape` in the storage, and returns a tensor handle for it.
    /// See [`ComputeClient::create_tensor`]
    pub fn empty_tensor(&self, shape: Shape, elem_size: usize) -> MemoryLayout {
        let descriptor =
            MemoryLayoutDescriptor::new(MemoryLayoutStrategy::Optimized, shape, elem_size);
        self.do_empty(vec![descriptor]).unwrap().remove(0)
    }

    /// Reserves all `shapes` in a single storage buffer, and returns the handles for them.
    /// See [`ComputeClient::create_tensor`]
    pub fn empty_tensors(&self, descriptors: Vec<MemoryLayoutDescriptor>) -> Vec<MemoryLayout> {
        self.do_empty(descriptors).unwrap()
    }

    /// Marks the given [Bytes] as being a staging buffer, maybe transferring it to pinned memory
    /// for faster data transfer with compute device.
    ///
    /// TODO: This blocks the compute queue, so it will drop the compute utilization.
    pub fn staging<'a, I>(&self, bytes: I, file_only: bool)
    where
        I: Iterator<Item = &'a mut Bytes>,
    {
        let has_staging = |b: &Bytes| match b.property() {
            AllocationProperty::Pinned => false,
            AllocationProperty::File => true,
            AllocationProperty::Native | AllocationProperty::Other => !file_only,
        };

        let mut to_be_updated = Vec::new();
        let sizes = bytes
            .filter_map(|b| match has_staging(b) {
                true => {
                    let len = b.len();
                    to_be_updated.push(b);
                    Some(len)
                }
                false => None,
            })
            .collect::<Vec<usize>>();

        if sizes.is_empty() {
            return;
        }

        let stream_id = self.stream_id();
        let sizes = sizes.to_vec();
        let stagings = self
            .device
            .submit_blocking(move |server| server.staging(&sizes, stream_id))
            .unwrap();

        let stagings = match stagings {
            Ok(val) => val,
            Err(_) => return,
        };

        to_be_updated
            .into_iter()
            .zip(stagings)
            .for_each(|(b, mut staging)| {
                b.copy_into(&mut staging);
                core::mem::swap(b, &mut staging);
            });
    }

    /// Transfer data from one client to another
    #[cfg_attr(
        feature = "runtime-tracing",
        tracing::instrument(level = "trace", skip(self, src, dst_server))
    )]
    pub fn to_client(&mut self, src: Handle, dst_server: &Self, dtype: ElemType) -> Handle {
        self.expect_local(&src.clone().binding());
        let shape = [src.size_in_used() as usize];
        let src_descriptor = src.copy_descriptor(shape.into(), [1].into(), 1);

        if R::Server::SERVER_COMM_ENABLED {
            self.to_client_tensor(src_descriptor, dst_server, dtype)
        } else {
            let alloc_desc = MemoryLayoutDescriptor::new(
                MemoryLayoutStrategy::Contiguous,
                src_descriptor.shape.clone(),
                src_descriptor.elem_size,
            );
            self.change_client_sync(src_descriptor, alloc_desc, dst_server)
                .memory
        }
    }

    /// Perform an `all_reduce` operation on the given devices.
    #[cfg_attr(
        feature = "runtime-tracing",
        tracing::instrument(level = "trace", skip(self, device_ids))
    )]
    pub fn ensure_init_collective(&mut self, device_ids: Vec<DeviceId>) {
        self.expect_device_transport(Collective::CommInit);
        let comm_id = CommunicationId::from(device_ids.clone());
        let is_comms_init = self
            .utilities
            .initialized_comms
            .read()
            .unwrap()
            .contains(&comm_id);
        if !is_comms_init {
            self.device
                .submit(move |server| server.comm_init(device_ids).unwrap());
            let mut initialized_comms = self.utilities.initialized_comms.write().unwrap();
            initialized_comms.insert(comm_id);
            // Flush immediately so other devices aren't blocked waiting on this initialization.
            self.device.flush_queue();
        }
    }

    /// Whether this runtime implements native inter-device transport.
    pub fn has_device_transport(&self) -> bool { R::Server::SERVER_COMM_ENABLED }

    #[track_caller]
    fn expect_device_transport(&self, operation: Collective) {
        if !self.has_device_transport() {
            panic!("{}", ServerError::no_device_transport(operation));
        }
    }

    /// Wait on the communication stream.
    #[cfg_attr(feature = "runtime-tracing", tracing::instrument(level = "trace", skip(self)))]
    pub fn sync_collective(&self) {
        if DeviceHandle::<R::Server>::is_blocking() {
            panic!("Can't use `sync_collective` with a blocking device handle");
        }
        if !self.has_device_transport() { return; }
        let stream_id = self.stream_id();

        self.device.submit(move |server| {
            if let Err(error) = server.sync_collective(stream_id) {
                log::error!("sync_collective failed: {error}");
            }
        });

        // We don't actually need or want to sync the server here, but we need to make sure any
        // task enqueued on the communication channel is done.
        self.device.flush_queue();
    }

    /// Perform an `all_reduce` operation on the given devices.
    #[cfg_attr(
        feature = "runtime-tracing",
        tracing::instrument(level = "trace", skip(self, src, dst, dtype, device_ids, op))
    )]
    pub fn all_reduce(
        &mut self,
        src: Handle,
        dst: Handle,
        dtype: ElemType,
        device_ids: Vec<DeviceId>,
        op: ReduceOperation,
    ) {
        if DeviceHandle::<R::Server>::is_blocking() {
            panic!("Can't use `all_reduce` with a blocking device handle");
        }
        self.expect_device_transport(Collective::AllReduce);

        let stream_id = self.stream_id();
        let src = src.binding();
        let dst = dst.binding();
        self.expect_local(&src);
        self.expect_local(&dst);

        self.ensure_init_collective(device_ids.clone());

        self.device.submit(move |server| {
            if let Err(error) = server.all_reduce(src, dst, dtype, stream_id, op, device_ids) {
                log::error!("all_reduce failed; the destination carries the failure: {error}");
            }
        });
    }

    /// Transfer data from one client to another
    ///
    /// Make sure the source description can be read in a contiguous manner.
    #[cfg_attr(
        feature = "runtime-tracing",
        tracing::instrument(level = "trace", skip(self, src_descriptor, dst_server))
    )]
    pub fn to_client_tensor(
        &mut self,
        src_descriptor: CopyDescriptor,
        dst_server: &Self,
        dtype: ElemType,
    ) -> Handle {
        self.expect_device_transport(Collective::Send);
        self.expect_local(&src_descriptor.handle);
        let stream_id_src = self.stream_id();
        let stream_id_dst = dst_server.stream_id();

        let device_id_src = self.device.device_id();
        let device_id_dst = dst_server.device.device_id();

        let mut dst_server = dst_server.clone();
        let handle = Handle::new(dst_server.service_id(), stream_id_dst, src_descriptor.handle.size_in_used());
        let handle_cloned = handle.clone();

        let device_ids = vec![device_id_src, device_id_dst];
        self.ensure_init_collective(device_ids.clone());
        dst_server.ensure_init_collective(device_ids);

        self.device.submit(move |server_src| {
            server_src
                .send(src_descriptor, dtype, stream_id_src, device_id_dst)
                .unwrap()
        });

        dst_server.device.submit(move |server_dst| {
            server_dst
                .recv(handle_cloned, dtype, stream_id_dst, device_id_src)
                .unwrap();
            server_dst.sync_collective(stream_id_dst).unwrap();
        });

        // `ServerCommunication::send` and`ServerCommunication::recv` are blocking: they each wait for the corresponding recv/send
        // call to be made. We flush the operations right away so that the neither server ends up in a deadlock.
        // The actual data transfer is still executed asynchronously on the communication stream.
        self.device.flush_queue();
        dst_server.device.flush_queue();

        handle
    }

    #[track_caller]
    #[cfg_attr(feature = "runtime-tracing", tracing::instrument(level="trace",
        skip(self, kernel, bindings),
        fields(
            kernel.name = %kernel.name(),
            kernel.id = %kernel.id(),
        )
    ))]
    unsafe fn launch_inner(
        &self,
        kernel: <R::Server as ComputeServer>::Kernel,
        count: RudaCount,
        bindings: KernelArguments,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) {
        if let RudaCount::Static(x, y, z) = &count
            && (*x == 0 || *y == 0 || *z == 0)
        {
            return;
        }
        if let RudaCount::Dynamic(binding) = &count {
            self.expect_local(binding);
        }
        for binding in &bindings.buffers {
            self.expect_local(binding);
        }
        for map in &bindings.tensor_maps {
            self.expect_local(&map.binding);
        }
        let launch_mode = crate::runtime::dry_run::launch_mode();
        if launch_mode.is_skipped() {
            self.device.submit(move |state| unsafe {
                state.launch_with_mode(kernel, count, bindings, mode, stream_id, launch_mode)
            });
            return;
        }
        let level = self.utilities.logger.profile_level();

        match level {
            None | Some(ProfileLevel::ExecutionOnly) => {
                let utilities = self.utilities.clone();
                self.device.submit(move |state| {
                    let name = kernel.name();
                    unsafe { state.launch_with_mode(kernel, count, bindings, mode, stream_id, launch_mode) };

                    if matches!(level, Some(ProfileLevel::ExecutionOnly)) {
                        let info = type_name_format(name, TypeNameFormatLevel::Balanced);
                        utilities.logger.register_execution(info);
                    }
                });
            }
            Some(level) => {
                let name = kernel.name();
                let kernel_id = kernel.id();
                let context = self.device.clone();
                let count_moved = count.clone();
                let (result, profile) = self
                    .profile(
                        move || {
                            context
                                .submit_blocking(move |state| unsafe {
                                    state.launch_with_mode(kernel, count_moved, bindings, mode, stream_id, launch_mode)
                                })
                                .unwrap()
                        },
                        name,
                    )
                    .unwrap();
                let info = match level {
                    ProfileLevel::Full => {
                        format!("{name}: {kernel_id} RudaCount {count:?}")
                    }
                    _ => type_name_format(name, TypeNameFormatLevel::Balanced),
                };
                self.utilities.logger.register_profiled(info, profile);
                result
            }
        }
    }

    /// Launches the `kernel` with the given `bindings`.
    #[track_caller]
    pub fn launch(
        &self,
        kernel: <R::Server as ComputeServer>::Kernel,
        count: RudaCount,
        bindings: KernelArguments,
    ) {
        // SAFETY: Using checked execution mode.
        unsafe {
            self.launch_inner(
                kernel,
                count,
                bindings,
                ExecutionMode::Checked,
                self.stream_id(),
            )
        }
    }

    /// Launches the `kernel` with the given `bindings` without performing any bound checks.
    ///
    /// # Safety
    ///
    /// To ensure this is safe, you must verify your kernel:
    /// - Has no out-of-bound reads and writes that can happen.
    /// - Has no infinite loops that might never terminate.
    #[track_caller]
    pub unsafe fn launch_unchecked(
        &self,
        kernel: <R::Server as ComputeServer>::Kernel,
        count: RudaCount,
        bindings: KernelArguments,
    ) {
        // SAFETY: Caller has to uphold kernel being safe.
        unsafe {
            self.launch_inner(
                kernel,
                count,
                bindings,
                match self.utilities.check_mode {
                    crate::runtime::config::compilation::BoundsCheckMode::Enforce => ExecutionMode::Checked,
                    crate::runtime::config::compilation::BoundsCheckMode::Validate => {
                        ExecutionMode::Validate
                    }
                    crate::runtime::config::compilation::BoundsCheckMode::Auto => ExecutionMode::Unchecked,
                },
                self.stream_id(),
            )
        }
    }

    /// Flush all outstanding commands.
    /// Submission succeeds or returns ServerError for the resolved stream.
    /// This is not a device-completion fence; wait on sync() when completion is required.
    pub fn flush(&self) -> Result<(), ServerError> {
        let stream_id = self.stream_id();

        self.device
            .submit_blocking(move |server| server.flush(stream_id))
            .unwrap()
    }

    /// Inspect enqueue-time failures carried by buffers, without a device barrier.
    pub fn check<'a>(&self, handles: impl IntoIterator<Item = &'a Handle>) -> Result<(), ServerError> {
        let bindings = self.bindings(handles)?;
        let stream = self.stream_id();
        self.device.submit_blocking(move |server| server.check(bindings, stream)).unwrap()
    }

    /// Wait for the buffers' work and inspect failures without a host copy.
    pub fn sync_buffers<'a>(
        &self,
        handles: impl IntoIterator<Item = &'a Handle>,
    ) -> DynFut<Result<(), ServerError>> {
        let bindings = match self.bindings(handles) {
            Ok(bindings) => bindings,
            Err(error) => return alloc::boxed::Box::pin(core::future::ready(Err(error))),
        };
        let stream = self.stream_id();
        let future = self.device.submit_blocking(move |server| server.sync_buffers(bindings, stream)).unwrap();
        self.utilities.logger.profile_summary();
        future
    }

    /// Request completion according to the server's resolved-stream contract.
    /// The returned future must be awaited or blocked on; merely obtaining it
    /// is not proof that device work has finished. Completion errors remain explicit.
    pub fn sync(&self) -> DynFut<Result<(), ServerError>> {
        let stream_id = self.stream_id();

        let fut = self
            .device
            .submit_blocking(move |server| server.sync(stream_id))
            .unwrap();

        self.utilities.logger.profile_summary();

        fut
    }

    /// Get the features supported by the compute server.
    pub fn properties(&self) -> &DeviceProperties {
        &self.utilities.properties
    }

    /// Get the features supported by the compute server.
    pub fn features(&self) -> &Features {
        &self.utilities.properties.features
    }

    /// # Warning
    ///
    /// For private use only.
    pub fn properties_mut(&mut self) -> Option<&mut DeviceProperties> {
        Arc::get_mut(&mut self.utilities).map(|state| &mut state.properties)
    }

    /// Get the current memory usage of this client.
    /// Returns allocator/server accounting, not process RSS, a predicted model
    /// capacity or a complete physical-GPU peak-memory measurement.
    pub fn memory_usage(&self) -> Result<MemoryUsage, ServerError> {
        let stream_id = self.stream_id();
        self.device
            .submit_blocking(move |server| server.memory_usage(stream_id))
            .unwrap()
    }

    /// Get all devices of a specific type available to this runtime
    pub fn enumerate_devices(&self, type_id: u16) -> Vec<DeviceId> {
        R::enumerate_devices(type_id, self.info())
    }

    /// Get all devices available to this runtime
    pub fn enumerate_all_devices(&self) -> Vec<DeviceId> {
        R::enumerate_all_devices(self.info())
    }

    /// Get the number of devices of a specific type available to this runtime
    pub fn device_count(&self, type_id: u16) -> usize {
        self.enumerate_devices(type_id).len()
    }

    /// Get the number of devices of a specific type available to this runtime
    pub fn device_count_total(&self) -> usize {
        self.enumerate_all_devices().len()
    }

    /// Change the memory allocation mode.
    ///
    /// # Safety
    ///
    /// This function isn't thread safe and might create memory leaks.
    pub unsafe fn allocation_mode(&self, mode: MemoryAllocationMode) {
        let stream_id = self.stream_id();
        self.device
            .submit(move |server| server.allocation_mode(mode, stream_id));
    }

    /// Ask the client to release memory that it can release.
    ///
    /// Nb: Results will vary on what the memory allocator deems beneficial,
    /// so it's not guaranteed any memory is freed.
    pub fn memory_cleanup(&self) {
        let stream_id = self.stream_id();
        self.device
            .submit(move |server| server.memory_cleanup(stream_id));
    }

    /// Explicitly compact adaptive memory into already held pages and release empty pages.
    /// Blocks until relocation copies complete and propagates errors. Pinned addresses stay fixed.
    /// Call outside graph capture. Fixed pools only release empty pages; no release is guaranteed.
    pub fn memory_compact(&self) -> Result<(), ServerError> {
        let stream_id = self.stream_id();
        self.device
            .submit_blocking(move |server| server.memory_compact(stream_id))
            .unwrap()
    }

    /// Open a profiling window without holding the device during the measured work.
    pub fn profile_start(&self) -> Result<ProfileWindow, ProfileError> {
        let stream_id = self.stream_id();
        let token = self.device.submit_blocking(move |server| server.start_profile(stream_id))
            .map_err(|error| ProfileError::Unknown { reason: format!("Profiling start call failed: {error:?}"),
                backtrace: BackTrace::capture() })?
            .map_err(|error| ProfileError::Server(alloc::boxed::Box::new(error)))?;
        Ok(ProfileWindow { stream_id, token })
    }

    /// Close on the window's original stream, regardless of the current caller's stream.
    pub fn profile_end(&self, window: ProfileWindow) -> Result<ProfileDuration, ProfileError> {
        self.device.submit_blocking(move |server| server.end_profile(window.stream_id, window.token))
            .map_err(|error| ProfileError::Unknown { reason: format!("Profiling end call failed: {error:?}"),
                backtrace: BackTrace::capture() })?
    }

    /// Abandon a window in queue order without running the remaining measured work.
    pub fn profile_abandon(&self, window: ProfileWindow) {
        self.device.submit(move |server| server.abandon_profile(window.stream_id, window.token));
    }

    /// Measure the execution time of some inner operations.
    #[track_caller]
    pub fn profile<O: Send + 'static>(
        &self,
        func: impl FnOnce() -> O + Send,
        #[allow(unused)] func_name: &str,
    ) -> Result<(O, ProfileDuration), ProfileError> {
        // Get the outer caller. For execute() this points straight to the
        // ruda kernel. For general profiling it points to whoever calls profile.
        #[cfg(feature = "runtime-profile-tracy")]
        let location = std::panic::Location::caller();

        // Make a CPU span. If the server has system profiling this is all you need.
        #[cfg(feature = "runtime-profile-tracy")]
        let _span = tracy_client::Client::running().unwrap().span_alloc(
            None,
            func_name,
            location.file(),
            location.line(),
            0,
        );

        let stream_id = self.stream_id();

        #[cfg(feature = "runtime-profile-tracy")]
        let gpu_span = if self.utilities.properties.timing_method == TimingMethod::Device {
            let gpu_span = self
                .utilities
                .gpu_client
                .span_alloc(func_name, "profile", location.file(), location.line())
                .unwrap();
            Some(gpu_span)
        } else {
            None
        };

        let device = self.device.clone();
        #[allow(unused_mut, reason = "Used in profile-tracy")]
        let mut result = self
            .device
            .exclusive(move || {
                // We first get mut access to the server to create a token.
                // Then we free to server, since it's going to be accessed in `func()`.
                let token =
                    match device.submit_blocking(move |server| server.start_profile(stream_id)) {
                        Ok(token) => match token {
                            Ok(token) => token,
                            Err(err) => return Err(err),
                        },
                        Err(err) => {
                            return Err(ServerError::Generic {
                                reason: alloc::format!(
                                    "Can't start profiling because of a call error: {err:?}"
                                ),
                                backtrace: BackTrace::capture(),
                            });
                        }
                    };

                let mut pending = PendingProfile { device: device.clone(), window: Some(ProfileWindow { stream_id, token }) };

                // We execute `func()` which will recursibly access the server.
                let out = func();

                // Finally we get the result from the token.
                let result = device
                    .submit_blocking(move |server| {
                        let mut result = server.end_profile(stream_id, token);

                        match result {
                            Ok(result) => Ok((out, result)),
                            Err(err) => Err(err),
                        }
                    })
                    .unwrap();

                pending.window.take();

                Ok(result)
            })
            .unwrap()
            .map_err(|err| ProfileError::Unknown {
                reason: alloc::format!("{err:?}"),
                backtrace: BackTrace::capture(),
            })?;

        #[cfg(feature = "runtime-profile-tracy")]
        if let Some(mut gpu_span) = gpu_span {
            gpu_span.end_zone();
            let epoch = self.utilities.epoch_time;
            // Add in the work to upload the timestamp data.
            result = result.map(|(o, result)| {
                (
                    o,
                    ProfileDuration::new(
                        alloc::boxed::Box::pin(async move {
                            let ticks = result.resolve().await;
                            if let Some(ticks) = &ticks {
                                let start_duration =
                                    ticks.start_duration_since(epoch).as_nanos() as i64;
                                let end_duration = ticks.end_duration_since(epoch).as_nanos() as i64;
                                gpu_span.upload_timestamp_start(start_duration);
                                gpu_span.upload_timestamp_end(end_duration);
                            }
                            ticks
                        }),
                        TimingMethod::Device,
                    ),
                )
            });
        }

        result
    }

    /// Transfer data from one client to another
    #[cfg_attr(
        feature = "runtime-tracing",
        tracing::instrument(
            level = "trace",
            skip(self, src_descriptor, alloc_descriptor, dst_server)
        )
    )]
    fn change_client_sync(
        &self,
        src_descriptor: CopyDescriptor,
        alloc_descriptor: MemoryLayoutDescriptor,
        dst_server: &Self,
    ) -> MemoryLayout {
        self.expect_local(&src_descriptor.handle);
        let shape = src_descriptor.shape.clone();
        let elem_size = src_descriptor.elem_size;
        let stream_id = self.stream_id();

        let read = self
            .device
            .submit_blocking(move |server| server.read(vec![src_descriptor], stream_id))
            .unwrap();

        let mut data = ruda_core::future::block_on(read).unwrap();

        let stream_id = dst_server.stream_id();
        let (handle_base, mut layouts) = dst_server
            .utilities
            .layout_policy
            .apply(dst_server.service_id(), stream_id, &[alloc_descriptor]);
        let alloc = layouts.remove(0);

        let desc_descriptor = CopyDescriptor {
            handle: alloc.memory.clone().binding(),
            shape,
            strides: alloc.strides.clone(),
            elem_size,
        };

        let (size, memory) = (handle_base.size(), handle_base.memory);
        dst_server.device.submit(move |server| {
            server.initialize_memory(memory, size, stream_id);
            server.write(vec![(desc_descriptor, data.remove(0))], stream_id)
        });

        alloc
    }

    /// Returns all vector sizes that are useful to perform optimal IO operation on the given element.
    pub fn io_optimized_vector_sizes(
        &self,
        size: usize,
    ) -> impl Iterator<Item = VectorSize> + Clone {
        self.properties().io_optimized_vector_sizes(size)
    }
}
