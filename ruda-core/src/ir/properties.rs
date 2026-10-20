use core::hash::{BuildHasher, Hash, Hasher};

use crate::ir::{
    AddressType, DeviceIdentity, SemanticType, StorageType, Type, TypeHash, VectorSize,
    features::{AtomicUsage, Features, Plane, TypeUsage},
};
use ruda_core::profile::TimingMethod;
use enumset::EnumSet;

/// Properties of the device related to the accelerator hardware.
///
/// # Plane size min/max
///
/// This is a range of possible values for the plane size.
///
/// For Nvidia GPUs and HIP, this is a single fixed value.
///
/// For wgpu with AMD GPUs this is a range of possible values, but the actual configured value
/// is undefined and can only be queried at runtime. Should usually be 32, but not guaranteed.
///
/// For Intel GPUs, this is variable based on the number of registers used in the kernel. No way to
/// query this at compile time is currently available. As a result, the minimum value should usually
/// be assumed.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HardwareProperties {
    /// The maximum size of a single load instruction, in bits. Used for optimized vector sizes.
    pub load_width: u32,
    /// Architectural vector registers, if the runtime can report a fixed register set.
    pub vector_register_count: Option<u32>,
    /// The minimum size of a plane on this device
    pub plane_size_min: u32,
    /// The maximum size of a plane on this device
    pub plane_size_max: u32,
    /// minimum number of bindings for a kernel that can be used at once.
    pub max_bindings: u32,
    /// Maximum amount of shared memory, in bytes
    pub max_shared_memory_size: usize,
    /// Maximum `RudaCount` in x, y and z dimensions
    pub max_ruda_count: (u32, u32, u32),
    /// Maximum number of total units in a ruda
    pub max_units_per_ruda: u32,
    /// Maximum `RudaDim` in x, y, and z dimensions
    pub max_ruda_dim: (u32, u32, u32),
    /// Number of streaming multiprocessors (SM), if available
    pub num_streaming_multiprocessors: Option<u32>,
    /// Number of available parallel cpu units, if the runtime is CPU.
    pub num_cpu_cores: Option<u32>,
    /// Bytes in the last-level cache, or `None` when the runtime cannot read it.
    pub last_level_cache_size: Option<usize>,
    /// Number of tensor cores per SM, if any
    pub num_tensor_cores: Option<u32>,
    /// The minimum tiling dimension for a single axis in tensor cores.
    ///
    /// For a backend that only supports 16x16x16, the value would be 16.
    /// For a backend that also supports 32x8x16, the value would be 8.
    pub min_tensor_cores_dim: Option<u32>,
    /// Maximum vector size supported by the device
    pub max_vector_size: VectorSize,
    /// Shared memory reserved by the driver for workgroup-scoped matrix operations.
    pub ruda_mma_reserved_shared_memory: usize,
}

/// Properties of the device related to allocation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct MemoryDeviceProperties {
    /// The maximum nr. of bytes that can be allocated in one go.
    pub max_page_size: u64,
    /// The required memory offset alignment in bytes.
    pub alignment: u64,
    max_memory: Option<u64>,
}

impl MemoryDeviceProperties {
    /// Allocation limits without a reported total memory budget.
    pub const fn new(max_page_size: u64, alignment: u64) -> Self {
        Self { max_page_size, alignment, max_memory: None }
    }

    /// The runtime's total memory budget, not the current free memory.
    ///
    /// This is distinct from the limit on one allocation. An unavailable budget is
    /// `None`, never a guessed multiple of `max_page_size` or `Some(0)`.
    pub const fn max_memory(&self) -> Option<u64> {
        self.max_memory
    }

    /// Set the reported budget, treating zero as unavailable.
    pub const fn with_max_memory(mut self, bytes: u64) -> Self {
        self.set_max_memory(bytes);
        self
    }

    /// Set the reported budget, treating zero as unavailable.
    pub const fn set_max_memory(&mut self, bytes: u64) {
        self.max_memory = match bytes {
            0 => None,
            bytes => Some(bytes),
        };
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorRegisters {
    count: usize,
    register_vector_size: VectorSize,
    max_vector_size: VectorSize,
}

impl VectorRegisters {
    pub fn new(hardware: &HardwareProperties, elem_size: usize) -> Option<Self> {
        Some(Self {
            count: hardware.vector_register_count? as usize,
            register_vector_size: hardware.vector_size_in(hardware.load_width, elem_size),
            max_vector_size: prev_power_of_two(hardware.max_vector_size.max(1)),
        })
    }

    pub fn count(&self) -> usize { self.count }

    pub fn registers_for(&self, vector_size: VectorSize) -> usize {
        vector_size.div_ceil(self.register_vector_size).max(1)
    }

    pub fn widest_vector_size(&self, live: usize) -> VectorSize {
        let registers = prev_power_of_two((self.count / live.max(1)).max(1));
        (registers * self.register_vector_size).min(self.max_vector_size)
    }

    pub fn vectors_fitting(&self, vector_size: VectorSize, reserved: usize) -> usize {
        self.count.saturating_sub(reserved) / self.registers_for(vector_size)
    }
}

impl HardwareProperties {
    fn vector_size_in(&self, width: u32, elem_size: usize) -> VectorSize {
        let elems = width as usize / (elem_size * 8);
        prev_power_of_two(elems.min(self.max_vector_size).max(1))
    }
}

fn prev_power_of_two(value: usize) -> usize {
    1 << value.ilog2()
}

fn vector_sizes_down_from(widest: usize) -> impl Iterator<Item = VectorSize> + Clone {
    (0..=widest.ilog2()).rev().map(|exponent| 1 << exponent)
}

/// Properties of what the device can do, like what `Feature` are
/// supported by it and what its memory properties are.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceProperties {
    /// The features supported by the runtime.
    pub features: Features,
    /// The memory properties of this client.
    pub memory: MemoryDeviceProperties,
    /// The topology properties of this client.
    pub hardware: HardwareProperties,
    /// The method used for profiling on the device.
    pub timing_method: TimingMethod,
    pub identity: DeviceIdentity,
}

impl TypeHash for DeviceProperties {
    fn write_hash(_hasher: &mut impl core::hash::Hasher) {
        // ignored.
    }
}

impl DeviceProperties {
    /// Create a new feature set with the given features and memory properties.
    pub fn new(
        features: Features,
        memory_props: MemoryDeviceProperties,
        hardware: HardwareProperties,
        timing_method: TimingMethod,
    ) -> Self {
        DeviceProperties {
            features,
            memory: memory_props,
            hardware,
            timing_method,
            identity: DeviceIdentity::default(),
        }
    }

    pub fn with_identity(mut self, identity: DeviceIdentity) -> Self {
        self.identity = identity;
        self
    }

    pub fn io_optimized_vector_sizes(
        &self,
        elem_size: usize,
    ) -> impl Iterator<Item = VectorSize> + Clone {
        vector_sizes_down_from(self.hardware.vector_size_in(self.hardware.load_width, elem_size))
    }

    pub fn vector_sizes_in_registers(
        &self,
        elem_size: usize,
        live: usize,
    ) -> impl Iterator<Item = VectorSize> + Clone {
        let widest = match VectorRegisters::new(&self.hardware, elem_size) {
            Some(registers) => registers.widest_vector_size(live),
            None => self.hardware.vector_size_in(self.hardware.load_width, elem_size),
        };
        vector_sizes_down_from(widest)
    }

    /// Upper bound on the number of planes in a ruda with `units` working units.
    pub fn max_plane_count(&self, units: u32) -> u32 {
        if self.features.plane.contains(Plane::Packed) && self.hardware.plane_size_min != 0 {
            units.div_ceil(self.hardware.plane_size_min)
        } else {
            units
        }
    }

    /// Get the usages for a type
    pub fn type_usage(&self, ty: StorageType) -> EnumSet<TypeUsage> {
        self.features.type_usage(ty)
    }

    /// Get the usages for an atomic type
    pub fn atomic_type_usage(&self, ty: Type) -> EnumSet<AtomicUsage> {
        self.features.atomic_type_usage(ty)
    }

    /// Whether the type is supported in any way
    pub fn supports_type(&self, ty: impl Into<Type>) -> bool {
        self.features.supports_type(ty)
    }

    /// Whether the address type is supported in any way
    pub fn supports_address(&self, ty: impl Into<AddressType>) -> bool {
        self.features.supports_address(ty)
    }

    /// Register an address type to the features
    pub fn register_address_type(&mut self, ty: impl Into<AddressType>) {
        self.features.types.address.insert(ty.into());
    }

    /// Register an address type to the features
    pub fn register_atomic_type_usage(&mut self, ty: Type, uses: impl Into<EnumSet<AtomicUsage>>) {
        *self.features.types.atomic.entry(ty).or_default() |= uses.into();
    }

    /// Register a storage type to the features
    pub fn register_type_usage(
        &mut self,
        ty: impl Into<StorageType>,
        uses: impl Into<EnumSet<TypeUsage>>,
    ) {
        *self.features.types.storage.entry(ty.into()).or_default() |= uses.into();
    }

    /// Register a semantic type to the features
    pub fn register_semantic_type(&mut self, ty: SemanticType) {
        self.features.types.semantic.insert(ty);
    }

    /// Create a stable hash of all device properties relevant to kernel compilation. Can be used
    /// as a stable checksum for a compilation cache.
    pub fn checksum(&self) -> u64 {
        let state = foldhash::fast::FixedState::default();
        let mut hasher = state.build_hasher();
        self.features.hash(&mut hasher);
        self.hardware.hash(&mut hasher);
        hasher.finish()
    }
}
