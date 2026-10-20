use ruda_core::device::{Device, DeviceId};

// It is not clear if AMD has a limit on the number of bindings it can hold at
// any given time, but it's highly unlikely that it's more than this. We can
// also assume that we'll never have more than this many bindings in flight,
// so it's 'safe' to store only this many bindings.
pub const AMD_MAX_BINDINGS: u32 = 1024;

#[derive(new, Clone, PartialEq, Eq, Default, Hash)]
pub struct AmdDevice {
    pub index: usize,
}

impl core::fmt::Debug for AmdDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("AmdDevice({})", self.index))
    }
}

impl Device for AmdDevice {
    fn from_id(device_id: DeviceId) -> Self {
        Self {
            index: device_id.index_id as usize,
        }
    }

    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: 0,
            index_id: self.index as u16,
        }
    }
}

impl AmdDevice {
    pub fn enumerate() -> Vec<Self> {
        <crate::HipRuntime as ruda_kernel::dsl::Runtime>::enumerate_all_devices(&())
            .into_iter().map(Self::from_id).collect()
    }

    pub fn find(&self) -> Result<(), usize> {
        let devices = Self::enumerate();
        if devices.iter().any(|device| device == self) { Ok(()) } else { Err(devices.len()) }
    }

    pub fn identity(&self) -> ruda_core::ir::DeviceIdentity {
        <crate::HipRuntime as ruda_kernel::dsl::Runtime>::client(self).properties().identity.clone()
    }
}
