use alloc::vec::Vec;
use ruda_core::device::Device;
use ruda_tensor::{DeviceId, DeviceOps, AutodiffContext};
use ruda_tensor::distributed::{CollectiveTensor, DistributedBackend, DistributedConfig,
    DistributedParams, ReduceOperation, TensorRef};
use ruda_autodiff::Autodiff;
use crate::{Dispatch, DispatchDevice, DispatchTensor, BackendTensor};

macro_rules! collective_device {
    ($device:expr, $call:ident) => {
        match $device.inner() {
            #[cfg(feature = "cuda")]
            DispatchDevice::Cuda(device) => $call!(crate::backends::Cuda, Cuda, device),
            #[cfg(feature = "tch")]
            DispatchDevice::LibTorch(device) => $call!(crate::backends::LibTorch, LibTorch, device),
            #[cfg(feature = "remote")]
            DispatchDevice::Remote(device) => $call!(crate::backends::Remote, Remote, device),
            device => panic!("Native collectives are unavailable on {device:?}"),
        }
    };
}

impl DistributedBackend for Dispatch {
    fn start_communication_server(devices: &[DispatchDevice], config: DistributedConfig) {
        let Some(first) = devices.first() else { return; };
        macro_rules! start {
            ($B:ty, $variant:ident, $device:expr) => {{
                let native = devices.iter().map(|device| match device.inner() {
                    DispatchDevice::$variant(device) => device.clone(),
                    _ => panic!("A collective group must use one native backend"),
                }).collect::<Vec<_>>();
                <$B>::start_communication_server(&native, config)
            }};
        }
        collective_device!(first, start)
    }

    fn close_communication_server(device: &DispatchDevice) {
        macro_rules! close {
            ($B:ty, $variant:ident, $device:expr) => { <$B>::close_communication_server($device) };
        }
        collective_device!(device, close)
    }

    fn register_sync_parameters(device: &DispatchDevice, params: Vec<DistributedParams>) {
        macro_rules! register {
            ($B:ty, $variant:ident, $device:expr) => { <$B>::register_sync_parameters($device, params) };
        }
        collective_device!(device, register)
    }

    fn submit_sync_collective(device: &DispatchDevice) {
        macro_rules! submit {
            ($B:ty, $variant:ident, $device:expr) => { <$B>::submit_sync_collective($device) };
        }
        collective_device!(device, submit)
    }

    fn submit_gradient_sync(tensor: TensorRef<Self>, params: DistributedParams) {
        // Only inspect the wrapper under its owning lock; the projection retains that owner.
        let device = unsafe { Self::comm_device(&tensor) };
        macro_rules! submit {
            ($B:ty, $variant:ident, $device:expr) => {{
                let native = tensor.map::<$B>(|tensor| match tensor {
                    DispatchTensor::$variant(BackendTensor::Float(value)) => value,
                    DispatchTensor::$variant(BackendTensor::AdFloat(value)) => &mut value.primitive,
                    _ => panic!("A gradient reference changed its backend or element kind"),
                });
                <$B>::submit_gradient_sync(native, params);
            }};
        }
        collective_device!(&device, submit)
    }

    fn all_reduce(tensor: DispatchTensor, op: ReduceOperation, device_ids: Vec<DeviceId>) -> CollectiveTensor<Self> {
        let device = tensor.device();
        macro_rules! reduce {
            ($B:ty, $variant:ident, $device:expr) => {{
                let native_ids = device_ids.into_iter().map(|id| match DispatchDevice::from_id(id).inner() {
                    DispatchDevice::$variant(device) => device.to_id(),
                    _ => panic!("A collective group must use one native backend"),
                }).collect();
                let value = match tensor {
                    DispatchTensor::$variant(BackendTensor::Float(value)) => {
                        let pending = <$B>::all_reduce(value, op, native_ids);
                        // The new wrapper remains unresolved until Dispatch::sync_collective.
                        BackendTensor::Float(unsafe { pending.assume_resolved() })
                    }
                    DispatchTensor::$variant(BackendTensor::AdFloat(value)) => {
                        let pending = Autodiff::<$B>::all_reduce(value, op, native_ids);
                        BackendTensor::AdFloat(unsafe { pending.assume_resolved() })
                    }
                    _ => panic!("All-reduce requires a floating-point tensor"),
                };
                CollectiveTensor::new(DispatchTensor::$variant(value))
            }};
        }
        collective_device!(&device, reduce)
    }

    fn sync_collective(device: &DispatchDevice) {
        macro_rules! sync {
            ($B:ty, $variant:ident, $device:expr) => { <$B>::sync_collective($device) };
        }
        collective_device!(device, sync)
    }
}

pub(crate) fn mark(tensor: DispatchTensor, param_id: ruda_tensor::distributed::DistributedParamId) -> DispatchTensor {
    assert!(matches!(tensor.context(), AutodiffContext::Enabled(_)), "Requires an autodiff tensor");
    match tensor {
        #[cfg(feature = "cuda")]
        DispatchTensor::Cuda(BackendTensor::AdFloat(value)) =>
            DispatchTensor::Cuda(BackendTensor::AdFloat(value.grad_distributed(param_id))),
        #[cfg(feature = "tch")]
        DispatchTensor::LibTorch(BackendTensor::AdFloat(value)) =>
            DispatchTensor::LibTorch(BackendTensor::AdFloat(value.grad_distributed(param_id))),
        #[cfg(feature = "remote")]
        DispatchTensor::Remote(BackendTensor::AdFloat(value)) =>
            DispatchTensor::Remote(BackendTensor::AdFloat(value.grad_distributed(param_id))),
        tensor => panic!("Distributed parameter registration is unavailable for {tensor:?}"),
    }
}

pub(crate) fn params(tensor: &DispatchTensor) -> Option<DistributedParams> {
    assert!(matches!(tensor.context(), AutodiffContext::Enabled(_)), "Requires an autodiff tensor");
    match tensor {
        #[cfg(feature = "cuda")]
        DispatchTensor::Cuda(BackendTensor::AdFloat(value)) =>
            <Autodiff<crate::backends::Cuda> as ruda_tensor::AutodiffBackend>::distributed_params(value),
        #[cfg(feature = "tch")]
        DispatchTensor::LibTorch(BackendTensor::AdFloat(value)) =>
            <Autodiff<crate::backends::LibTorch> as ruda_tensor::AutodiffBackend>::distributed_params(value),
        #[cfg(feature = "remote")]
        DispatchTensor::Remote(BackendTensor::AdFloat(value)) =>
            <Autodiff<crate::backends::Remote> as ruda_tensor::AutodiffBackend>::distributed_params(value),
        _ => None,
    }
}
