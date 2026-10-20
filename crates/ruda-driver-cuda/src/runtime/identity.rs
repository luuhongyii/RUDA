use cudarc::driver::sys::*;
use ruda_core::ir::{DeviceIdentity, PciVendor, PhysicalDevice};
use std::ffi::CStr;

pub(super) fn probe(device: CUdevice) -> DeviceIdentity {
    let name = cudarc::driver::result::device::get_name(device)
        .unwrap_or_else(|_| "unknown CUDA device".into());
    let mut physical = PhysicalDevice::default();
    physical.vendor = Some(PciVendor::Nvidia);
    let mut bus_id = [0u8; 32];
    let status = unsafe { cuDeviceGetPCIBusId(bus_id.as_mut_ptr().cast(), bus_id.len() as _, device) };
    if status == CUresult::CUDA_SUCCESS {
        physical.pci_address = CStr::from_bytes_until_nul(&bus_id).ok()
            .and_then(|id| id.to_str().ok()?.parse().ok());
    }
    #[cfg(windows)]
    {
        let mut luid = [0 as core::ffi::c_char; 8];
        let mut node_mask = 0;
        let status = unsafe { cuDeviceGetLuid(luid.as_mut_ptr(), &mut node_mask, device) };
        if status == CUresult::CUDA_SUCCESS {
            physical.luid = Some(ruda_core::ir::AdapterLuid::new(luid.map(|byte| byte as u8)));
        }
    }
    DeviceIdentity { name, fingerprint: String::new(), physical: Some(physical) }
}
