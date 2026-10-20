use crate::{MetalDevice, MetalServer};
use objc2_metal::{MTLDevice, MTLGPUFamily};
use ruda::runtime::{
    allocator::ContiguousMemoryLayoutPolicy, backend::Runtime, client::ComputeClient,
    logging::ServerLogger, memory_management::MemoryConfiguration,
    server::{ComputeServer, ServerUtilities},
};
use ruda_compiler::cpp::{
    DialectWmmaCompiler, metal::{MslDialect, arch::MetalArchitecture},
    shared::register_wmma_features,
};
use ruda_core::{
    device::{Device, DeviceId, DeviceService, ServerUtilitiesHandle, ServiceId},
    ir::{
        AddressType, DeviceIdentity, DeviceProperties, ElemType, FloatKind, HardwareProperties,
        IntKind, MemoryDeviceProperties, PhysicalDevice, RegistryEntryId, TargetProperties,
        StorageType, Type, UIntKind, features::{AtomicUsage, Plane, TypeUsage},
    },
    profile::TimingMethod,
    tensor::{Shape, Strides, striding::has_pitched_row_major_strides},
};
use std::sync::Arc;

pub type MetalCompiler = ruda_kernel::dsl::lowering::cpp::MslCompiler;

#[derive(Debug, Clone)]
pub struct MetalRuntime;

impl Runtime for MetalRuntime {
    type Compiler = MetalCompiler;
    type Server = MetalServer;
    type Device = MetalDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self> { ComputeClient::load(device) }
    fn name(_: &ComputeClient<Self>) -> &'static str { "metal" }
    fn max_ruda_count() -> (u32, u32, u32) { (u32::MAX, u32::MAX, u32::MAX) }
    fn can_read_tensor(shape: &Shape, strides: &Strides) -> bool {
        shape.is_empty() || has_pitched_row_major_strides(shape, strides)
    }
    fn target_properties() -> TargetProperties { TargetProperties::default() }
    fn enumerate_devices(type_id: u16, _: &()) -> Vec<DeviceId> {
        match type_id {
            0 => crate::device::all_devices().first().map(|_| DeviceId { type_id: 0, index_id: 0 })
                .into_iter().collect(),
            1 | 2 => crate::device::all_devices().into_iter()
                .filter(|device| device.hasUnifiedMemory() == (type_id == 2))
                .enumerate().map(|(index, _)| DeviceId {
                    type_id, index_id: u16::try_from(index).expect("Metal device index exceeds DeviceId"),
                }).collect(),
            _ => Vec::new(),
        }
    }
    fn enumerate_all_devices(info: &()) -> Vec<DeviceId> {
        [1, 2].into_iter().flat_map(|kind| Self::enumerate_devices(kind, info)).collect()
    }
    fn utilization(device: &Self::Device) -> Result<
        ruda::runtime::backend::DeviceUtilization, ruda::runtime::backend::UtilizationUnavailable,
    > {
        ruda::runtime::monitoring::gpu_utilization::CardCounters::read(
            Self::client(device).properties().identity.physical.as_ref(),
        )
    }
}

impl DeviceService for MetalServer {
    fn init(device_id: DeviceId) -> Self {
        let device = crate::device::resolve(MetalDevice::from_id(device_id));
        assert!(device.supportsFamily(MTLGPUFamily::Metal3),
            "Ruda native Metal requires Metal 3 or later: {}", device.name());
        let memory = MemoryDeviceProperties::new(device.maxBufferLength() as u64, 256)
            .with_max_memory(device.recommendedMaxWorkingSetSize());
        let dim = device.maxThreadsPerThreadgroup();
        let hardware = HardwareProperties {
            load_width: 128, vector_register_count: None, plane_size_min: 32, plane_size_max: 32,
            max_bindings: 30, max_shared_memory_size: device.maxThreadgroupMemoryLength(),
            max_ruda_count: (u32::MAX, u32::MAX, u32::MAX),
            max_units_per_ruda: dim.width as u32,
            max_ruda_dim: (dim.width as u32, dim.height as u32, dim.depth as u32),
            num_streaming_multiprocessors: None, num_tensor_cores: None,
            min_tensor_cores_dim: None, num_cpu_cores: None, last_level_cache_size: None,
            max_vector_size: 4, ruda_mma_reserved_shared_memory: 0,
        };
        let name = device.name().to_string();
        let mut physical = PhysicalDevice::default();
        physical.registry_entry_id = Some(RegistryEntryId::new(device.registryID()));
        let mut properties = DeviceProperties::new(Default::default(), memory.clone(), hardware, TimingMethod::Device)
            .with_identity(DeviceIdentity { fingerprint: format!("msl_{name}"), name, physical: Some(physical) });
        register_features(&mut properties);
        let mut utilities = ServerUtilities::new(
            ServiceId::of::<Self>(device_id), properties, Arc::new(ServerLogger::default()), (),
            ContiguousMemoryLayoutPolicy::new(memory.alignment as usize),
        );
        utilities.runtime_name = Some("metal");
        MetalServer::new(device, memory, MemoryConfiguration::default(), utilities)
    }

    fn utilities(&self) -> ServerUtilitiesHandle { ComputeServer::utilities(self) as ServerUtilitiesHandle }
}

fn register_features(properties: &mut DeviceProperties) {
    properties.register_address_type(AddressType::U32);
    properties.register_address_type(AddressType::U64);
    for ty in [
        ElemType::UInt(UIntKind::U8), ElemType::UInt(UIntKind::U16),
        ElemType::UInt(UIntKind::U32), ElemType::UInt(UIntKind::U64),
        ElemType::Int(IntKind::I8), ElemType::Int(IntKind::I16),
        ElemType::Int(IntKind::I32), ElemType::Int(IntKind::I64),
        ElemType::Float(FloatKind::F16), ElemType::Float(FloatKind::BF16),
        ElemType::Float(FloatKind::F32), ElemType::Bool,
    ] { properties.register_type_usage(ty, TypeUsage::all()); }
    for ty in [FloatKind::E4M3, FloatKind::E5M2] {
        properties.register_type_usage(ElemType::Float(ty), TypeUsage::Conversion | TypeUsage::Buffer);
    }
    for ty in [ElemType::Int(IntKind::I32), ElemType::UInt(UIntKind::U32)] {
        properties.register_atomic_type_usage(Type::new(StorageType::Atomic(ty)), AtomicUsage::all());
    }
    properties.register_atomic_type_usage(
        Type::new(StorageType::Atomic(ElemType::Float(FloatKind::F32))), AtomicUsage::Add | AtomicUsage::LoadStore,
    );
    register_wmma_features(MslDialect::supported_wmma_combinations(&MetalArchitecture::Metal3), properties);
    properties.features.alignment = true;
    properties.features.device_memory_scope = true;
    properties.features.plane.insert(Plane::Ops);
    properties.features.plane.insert(Plane::Sync);
    properties.features.plane.insert(Plane::NonUniformControlFlow);
}

