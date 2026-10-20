use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};
use std::{collections::HashMap, sync::{Mutex, OnceLock}};

pub use ruda::runtime::device::MetalDevice;

struct Registry {
    next: u32,
    devices: HashMap<u32, Retained<ProtocolObject<dyn MTLDevice>>>,
}

// Metal devices may be retained and queried from any thread. Registry mutation is locked.
unsafe impl Send for Registry {}

fn registry() -> &'static Mutex<Registry> {
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Registry { next: 0, devices: HashMap::new() }))
}

/// Register a native device for use by the runtime.
pub fn register_device(device: Retained<ProtocolObject<dyn MTLDevice>>) -> MetalDevice {
    let mut registry = registry().lock().unwrap_or_else(|error| error.into_inner());
    let id = registry.next;
    assert!(id <= u16::MAX as u32, "Metal device registry exhausted DeviceId indices");
    registry.next += 1;
    registry.devices.insert(id, device);
    MetalDevice::Existing(id)
}

pub(crate) fn all_devices() -> Vec<Retained<ProtocolObject<dyn MTLDevice>>> {
    #[cfg(target_os = "macos")]
    { objc2_metal::MTLCopyAllDevices().to_vec() }
    #[cfg(not(target_os = "macos"))]
    { MTLCreateSystemDefaultDevice().into_iter().collect() }
}

pub(crate) fn resolve(device: MetalDevice) -> Retained<ProtocolObject<dyn MTLDevice>> {
    match device {
        MetalDevice::DefaultDevice => MTLCreateSystemDefaultDevice()
            .or_else(|| all_devices().into_iter().next()).expect("No Metal device available"),
        MetalDevice::DiscreteGpu(index) => all_devices().into_iter()
            .filter(|device| !device.hasUnifiedMemory()).nth(index).expect("Discrete Metal GPU not found"),
        MetalDevice::IntegratedGpu(index) => all_devices().into_iter()
            .filter(|device| device.hasUnifiedMemory()).nth(index).expect("Integrated Metal GPU not found"),
        MetalDevice::Existing(id) => registry().lock().unwrap_or_else(|error| error.into_inner())
            .devices.get(&id).cloned().expect("Metal device is not registered"),
    }
}

