use alloc::{boxed::Box, sync::Arc, vec::Vec};
use core::marker::PhantomData;

use crate::dsl::Runtime;
use crate::dsl::prelude::{ArrayArg, LaunchMarker, TensorArg, TensorMapArg, TensorMapKind};
use crate::dsl::{InfoBuilder, KernelSettings, ScalarArgType};
#[cfg(any(feature = "frontend-std", feature = "std"))]
use core::cell::RefCell;
use ruda_core::ir::{AddressType, DeviceProperties, Scope, StorageType, TargetProperties, Type};
use ruda_core::kernel::{BufferIOAttr, KernelIo};
use ruda::runtime::server::{Binding, RudaCount, TensorMapBinding};
use ruda::runtime::{
    client::{Client, ComputeClient},
    compiler::RudaTask,
    kernel::{RudaKernel, KernelTask},
    server::{KernelArguments, ServerError},
};

#[cfg(any(feature = "frontend-std", feature = "std"))]
std::thread_local! {
    static INFO: RefCell<InfoBuilder> = RefCell::new(InfoBuilder::default());
    // Only used for resolving types
    static SCOPE: RefCell<Scope> = RefCell::new(Scope::root(false));
}

/// Compilation properties from a live client or an offline target description.
#[derive(Clone)]
pub enum KernelContext {
    /// Read properties from the actual selected device service.
    Client(Client),
    /// Expand a kernel without initializing or submitting to a device.
    Properties {
        /// Capabilities of the target device.
        device: Arc<DeviceProperties>,
        /// Instruction-level properties of its compiler target.
        target: Arc<TargetProperties>,
    },
}

impl KernelContext {
    /// Create an offline expansion context from explicit target properties.
    pub fn from_properties(device: Arc<DeviceProperties>, target: Arc<TargetProperties>) -> Self {
        Self::Properties { device, target }
    }

    /// Capabilities used when registering and expanding arguments.
    pub fn properties(&self) -> &DeviceProperties {
        match self {
            Self::Client(client) => client.properties(),
            Self::Properties { device, .. } => device,
        }
    }

    /// Instruction properties used when expanding the kernel body.
    pub fn target_properties(&self) -> TargetProperties {
        match self {
            Self::Client(client) => client.target_properties(),
            Self::Properties { target, .. } => target.as_ref().clone(),
        }
    }
}

impl From<Client> for KernelContext {
    fn from(client: Client) -> Self { Self::Client(client) }
}

impl<R: Runtime> From<ComputeClient<R>> for KernelContext {
    fn from(client: ComputeClient<R>) -> Self { Self::Client(client.erase()) }
}

/// A fully registered kernel invocation, not compiled or submitted yet.
///
/// Buffers, scalars and shape metadata are owned. The execution queue is fixed
/// at preparation time. This is an opt-in building block for native graph
/// backends, not stream capture or a CPU implementation of a GPU kernel.
pub struct PreparedKernel<R: Runtime> {
    task: Box<dyn RudaTask<R::Compiler>>,
    count: RudaCount,
    arguments: KernelArguments,
    client: ComputeClient<R>,
    scalar_words: usize,
}

impl<R: Runtime> PreparedKernel<R> {
    /// Number of aligned u64 words in the packed scalar prefix. Backend graph
    /// updates must keep the remaining (shape/stride/length) metadata unchanged.
    pub fn scalar_words(&self) -> usize { self.scalar_words }

    /// Consume the prepared invocation. Backends must honor the client's device
    /// and queue, retain arguments, and validate any restrictions before launch.
    pub fn into_parts(self) -> (
        Box<dyn RudaTask<R::Compiler>>, RudaCount, KernelArguments, ComputeClient<R>,
    ) {
        (self.task, self.count, self.arguments, self.client)
    }
}

/// A prepared DSL invocation with no selected Rust runtime or compiler type.
pub struct ErasedPreparedKernel {
    task: Box<dyn RudaKernel>,
    count: RudaCount,
    arguments: KernelArguments,
    client: Client,
    scalar_words: usize,
}

impl ErasedPreparedKernel {
    /// Length of the packed scalar prefix, excluding shape/stride metadata.
    pub fn scalar_words(&self) -> usize { self.scalar_words }

    /// Consume the invocation while retaining its actual service and fixed queue.
    pub fn into_parts(self) -> (Box<dyn RudaKernel>, RudaCount, KernelArguments, Client) {
        (self.task, self.count, self.arguments, self.client)
    }

    /// Adapt a prepared invocation to a native graph API after checking its service type.
    pub fn into_typed<R: Runtime>(self) -> Result<PreparedKernel<R>, ServerError> {
        let client = self.client.typed_service::<R>()?;
        Ok(PreparedKernel {
            task: Box::new(KernelTask::<R::Compiler, _>::new(self.task)),
            count: self.count,
            arguments: self.arguments,
            client,
            scalar_words: self.scalar_words,
        })
    }

    /// Submit the invocation with bounds checking on its captured queue.
    #[track_caller]
    pub fn launch(self) { self.client.launch(self.task, self.count, self.arguments); }

    /// Submit without bounds checking on its captured queue.
    ///
    /// # Safety
    /// The kernel must have valid memory accesses and terminate.
    #[track_caller]
    pub unsafe fn launch_unchecked(self) {
        unsafe { self.client.launch_unchecked(self.task, self.count, self.arguments); }
    }
}

/// Prepare a kernel for [launch](KernelLauncher::launch).
pub struct KernelLauncher<R: LaunchMarker = ()> {
    buffers: Vec<Binding>,
    tensor_maps: Vec<TensorMapBinding>,
    declared_io: KernelIo,
    declaring: BufferIOAttr,
    address_type: AddressType,
    pub settings: KernelSettings,
    #[cfg(not(any(feature = "frontend-std", feature = "std")))]
    info: InfoBuilder,
    #[cfg(not(any(feature = "frontend-std", feature = "std")))]
    pub scope: Scope,
    _runtime: PhantomData<R>,
}

impl<R: LaunchMarker> KernelLauncher<R> {
    #[cfg(any(feature = "frontend-std", feature = "std"))]
    pub fn with_scope<T>(&mut self, fun: impl FnMut(&mut Scope) -> T) -> T {
        SCOPE.with_borrow_mut(fun)
    }

    #[cfg(not(any(feature = "frontend-std", feature = "std")))]
    pub fn with_scope<T>(&mut self, mut fun: impl FnMut(&mut Scope) -> T) -> T {
        fun(&mut self.scope)
    }

    #[cfg(any(feature = "frontend-std", feature = "std"))]
    fn with_info<T>(&mut self, fun: impl FnMut(&mut InfoBuilder) -> T) -> T {
        INFO.with_borrow_mut(fun)
    }

    #[cfg(not(any(feature = "frontend-std", feature = "std")))]
    fn with_info<T>(&mut self, mut fun: impl FnMut(&mut InfoBuilder) -> T) -> T {
        fun(&mut self.info)
    }

    /// Register a scalar to be launched.
    pub fn register_scalar<C: ScalarArgType>(&mut self, scalar: C) {
        self.with_info(|info| info.scalars.push(scalar));
    }

    /// Register a scalar to be launched from raw data.
    pub fn register_scalar_raw(&mut self, bytes: &[u8], dtype: StorageType) {
        self.with_info(|info| info.scalars.push_raw(bytes, dtype));
    }

    /// Finish argument registration without submitting any GPU computation.
    /// Construct each launcher and consume it before preparing the next one:
    /// scalar/metadata registration uses the existing thread-local builder.
    pub fn prepare<K: RudaKernel>(
        mut self, count: RudaCount, kernel: K, client: &ComputeClient<R>,
    ) -> PreparedKernel<R> where R: Runtime {
        let scalar_words = self.with_info(|info| info.scalars.len_aligned());
        PreparedKernel {
            scalar_words,
            arguments: self.into_bindings(),
            task: Box::new(KernelTask::<R::Compiler, K>::new(kernel)),
            count,
            client: client.fixed_execution_queue(),
        }
    }

    /// Launch the kernel.
    #[track_caller]
    pub fn launch<K: RudaKernel>(
        self,
        ruda_count: RudaCount,
        kernel: K,
        client: &ComputeClient<R>,
    ) where R: Runtime {
        let bindings = self.into_bindings();
        let kernel = Box::new(KernelTask::<R::Compiler, K>::new(kernel));

        client.launch(kernel, ruda_count, bindings)
    }

    /// Drain scalar/metadata registration for an invocation that will not be launched.
    pub fn discard(self) {
        let _ = self.into_bindings();
    }

    /// Launch the kernel without check bounds.
    ///
    /// # Safety
    ///
    /// The kernel must not:
    /// - Contain any out of bounds reads or writes. Doing so is immediate UB.
    /// - Contain any loops that never terminate. These may be optimized away entirely or cause
    ///   other unpredictable behaviour.
    #[track_caller]
    pub unsafe fn launch_unchecked<K: RudaKernel>(
        self,
        ruda_count: RudaCount,
        kernel: K,
        client: &ComputeClient<R>,
    ) where R: Runtime {
        unsafe {
            let bindings = self.into_bindings();
            let kernel = Box::new(KernelTask::<R::Compiler, K>::new(kernel));

            client.launch_unchecked(kernel, ruda_count, bindings)
        }
    }

    /// Register an owned invocation for the selected client without compiling it.
    pub fn prepare_erased<K: RudaKernel>(
        mut self, count: RudaCount, kernel: K, client: &Client,
    ) -> ErasedPreparedKernel {
        let scalar_words = self.with_info(|info| info.scalars.len_aligned());
        ErasedPreparedKernel {
            scalar_words,
            arguments: self.into_bindings(),
            task: Box::new(kernel),
            count,
            client: client.fixed_execution_queue(),
        }
    }

    /// Launch directly through the selected device, without a Rust runtime type.
    #[track_caller]
    pub fn launch_erased<K: RudaKernel>(self, count: RudaCount, kernel: K, client: &Client) {
        client.launch(Box::new(kernel), count, self.into_bindings());
    }

    /// Launch directly through the selected device without bounds checking.
    ///
    /// # Safety
    /// The kernel must have valid memory accesses and terminate.
    #[track_caller]
    pub unsafe fn launch_unchecked_erased<K: RudaKernel>(
        self, count: RudaCount, kernel: K, client: &Client,
    ) {
        unsafe { client.launch_unchecked(Box::new(kernel), count, self.into_bindings()); }
    }

    /// We need to create the bindings in the same order they are defined in the compilation step.
    ///
    /// The function [`crate::dsl::KernelIntegrator::integrate`] stars by registering the input tensors followed
    /// by the output tensors. Then the tensor metadata, and the scalars at the end. The scalars
    /// are registered in the same order they are added. This is why we store the scalar data type
    /// in the `scalar_order` vector, so that we can register them in the same order.
    ///
    /// Also returns an ordered list of constant bindings. The ordering between constants and tensors
    /// is up to the runtime.
    fn into_bindings(mut self) -> KernelArguments {
        let mut bindings = KernelArguments::new();
        let address_type = self.address_type;
        let info = self.with_info(|info| info.finish(address_type));

        bindings.buffers = self.buffers;
        bindings.tensor_maps = self.tensor_maps;
        bindings.declared_io = self.declared_io;
        bindings.info = info;

        bindings
    }
}

// Tensors/arrays
impl<R: LaunchMarker> KernelLauncher<R> {
    /// Declare accesses for resources registered until the next declaration.
    /// Generated launches set this per argument; manual launchers default to read-write.
    pub fn declare_io(&mut self, io: BufferIOAttr) {
        self.declaring = io;
    }

    fn alias_io(&mut self, input_pos: usize) {
        if self.declaring.writes()
            && let Some(io) = self.declared_io.buffers.get_mut(input_pos)
        {
            *io = BufferIOAttr::ReadWrite;
        }
    }

    /// Push a new input tensor to the state.
    pub fn register_tensor(&mut self, tensor: TensorArg<R>, ty: Type) {
        if let Some(tensor) = self.process_tensor(tensor, ty) {
            self.declared_io.buffers.push(self.declaring);
            self.buffers.push(tensor);
        }
    }

    fn process_tensor(&mut self, tensor: TensorArg<R>, ty: Type) -> Option<Binding> {
        let tensor = match tensor {
            TensorArg::Handle { handle, .. } => handle,
            TensorArg::Alias { input_pos, .. } => {
                self.alias_io(input_pos);
                return None;
            }
        };

        let elem_size = ty.size();
        let vectorization = ty.vector_size();

        let buffer_len = tensor.handle.size_in_used() / elem_size as u64;
        let len = tensor.shape.iter().product::<usize>() / vectorization;
        let address_type = self.address_type;
        self.with_info(|info| {
            info.metadata.register_tensor(
                tensor.strides.len() as u64,
                buffer_len,
                len as u64,
                tensor.shape.clone(),
                tensor.strides.clone(),
                address_type,
            )
        });
        Some(tensor.handle)
    }

    /// Push a new input array to the state.
    pub fn register_array(&mut self, array: ArrayArg<R>, ty: Type) {
        if let Some(tensor) = self.process_array(array, ty) {
            self.declared_io.buffers.push(self.declaring);
            self.buffers.push(tensor);
        }
    }

    fn process_array(&mut self, array: ArrayArg<R>, ty: Type) -> Option<Binding> {
        let array = match array {
            ArrayArg::Handle { handle, .. } => handle,
            ArrayArg::Alias { input_pos, .. } => {
                self.alias_io(input_pos);
                return None;
            }
        };

        let elem_size = ty.size();
        let vectorization = ty.vector_size();

        let buffer_len = array.handle.size_in_used() / elem_size as u64;
        let address_type = self.address_type;
        self.with_info(|info| {
            info.metadata.register_array(
                buffer_len,
                array.length[0] as u64 / vectorization as u64,
                address_type,
            )
        });
        Some(array.handle)
    }

    /// Push a new tensor to the state.
    pub fn register_tensor_map<K: TensorMapKind>(&mut self, map: TensorMapArg<R, K>, ty: Type) {
        let binding = self
            .process_tensor(map.tensor, ty)
            .expect("Can't use alias for TensorMap");

        let map = map.metadata.clone();
        // TMA may store through an immutable map reference; the signature cannot prove read-only.
        self.declared_io.tensor_maps.push(BufferIOAttr::ReadWrite);
        self.tensor_maps.push(TensorMapBinding { binding, map });
    }
}

impl<R: LaunchMarker> KernelLauncher<R> {
    pub fn new(settings: KernelSettings) -> Self {
        Self {
            address_type: settings.address_type,
            settings,
            buffers: Vec::new(),
            tensor_maps: Vec::new(),
            declared_io: KernelIo::default(),
            declaring: BufferIOAttr::ReadWrite,
            _runtime: PhantomData,
            #[cfg(not(any(feature = "frontend-std", feature = "std")))]
            info: InfoBuilder::default(),
            #[cfg(not(any(feature = "frontend-std", feature = "std")))]
            scope: Scope::root(false),
        }
    }
}
