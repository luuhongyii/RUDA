use ruda_core::device::{Device, DeviceId};

#[derive(new, Clone, PartialEq, Eq, Default, Hash, Debug)]
pub struct CpuDevice;

impl Device for CpuDevice {
    fn from_id(_device_id: DeviceId) -> Self {
        Self
    }

    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: 0,
            index_id: 0,
        }
    }
}

impl CpuDevice {
    pub fn enumerate() -> Vec<Self> {
        <crate::CpuRuntime as ruda_kernel::dsl::Runtime>::enumerate_all_devices(&())
            .into_iter().map(Self::from_id).collect()
    }

    pub fn find(&self) -> Result<(), usize> { Ok(()) }

    pub fn identity(&self) -> ruda_core::ir::DeviceIdentity {
        <crate::CpuRuntime as ruda_kernel::dsl::Runtime>::client(self).properties().identity.clone()
    }
}
