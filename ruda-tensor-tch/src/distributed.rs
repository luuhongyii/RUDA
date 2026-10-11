//! LibTorch tensor/stream adapter for ruCCL's native in-process collectives.
use crate::{LibTorch, LibTorchDevice, TchElement, TchTensor};
use ruccl::in_process::native::{NativeCollectiveDevice, NativeCommunicators};
use ruda_tensor::{
    DeviceId, TensorMetadata,
    distributed::{CollectiveTensor, DistributedBackend, ReduceOperation},
};
use std::{
    error::Error,
    ffi::{CStr, c_char, c_void},
    fmt::{Display, Formatter},
    ptr::NonNull,
    sync::OnceLock,
};

unsafe extern "C" {
    fn ruda_tch_nccl_available() -> i32;
    fn ruda_tch_prepare_collective(tensor: *const c_void) -> *mut c_void;
    fn ruda_tch_drop_prepared(tensor: *mut c_void);
    fn ruda_tch_all_reduce(
        inputs: *const *const c_void,
        count: usize,
        mean: i32,
        outputs: *mut *mut c_void,
    ) -> i32;
    fn ruda_tch_collective_error() -> *const c_char;
}

/// Whether this build supplies ruCCL with a CUDA/NCCL device execution adapter.
pub fn nccl_available() -> bool {
    unsafe { ruda_tch_nccl_available() != 0 }
}

#[derive(Debug)]
struct DeviceError(String);
impl Display for DeviceError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}
impl Error for DeviceError {}
fn ffi_error() -> DeviceError {
    DeviceError(
        unsafe { CStr::from_ptr(ruda_tch_collective_error()) }
            .to_string_lossy()
            .into_owned(),
    )
}

struct PreparedTensor {
    handle: NonNull<c_void>,
    tensor: TchTensor,
}
// The owned C++ tensor and captured CUDA stream are only used by the ruCCL
// generation executor; completion precedes release of either tensor reference.
unsafe impl Send for PreparedTensor {}
impl Drop for PreparedTensor {
    fn drop(&mut self) {
        unsafe {
            ruda_tch_drop_prepared(self.handle.as_ptr());
        }
    }
}

#[derive(Clone)]
struct TorchDevice;
impl NativeCollectiveDevice for TorchDevice {
    type Tensor = TchTensor;
    type Prepared = PreparedTensor;
    type Error = DeviceError;

    fn prepare(&self, tensor: TchTensor, device: DeviceId) -> Result<PreparedTensor, DeviceError> {
        if device.type_id != 0
            || tensor.tensor.device() != tch::Device::Cuda(device.index_id as usize)
        {
            return Err(DeviceError(
                "ruCCL LibTorch adapter requires this rank's CUDA tensor".into(),
            ));
        }
        let handle =
            NonNull::new(unsafe { ruda_tch_prepare_collective(tensor.tensor.as_ptr().cast()) })
                .ok_or_else(ffi_error)?;
        Ok(PreparedTensor { handle, tensor })
    }

    fn all_reduce(
        &self,
        devices: &[DeviceId],
        inputs: &[PreparedTensor],
        operation: ReduceOperation,
    ) -> Result<Vec<TchTensor>, DeviceError> {
        if inputs.is_empty() || inputs.len() != devices.len() {
            return Err(DeviceError(
                "ruCCL device world and prepared input count differ".into(),
            ));
        }
        let pointers = inputs
            .iter()
            .map(|input| input.handle.as_ptr().cast_const())
            .collect::<Vec<_>>();
        let mut outputs = vec![std::ptr::null_mut::<c_void>(); inputs.len()];
        let result = unsafe {
            ruda_tch_all_reduce(
                pointers.as_ptr(),
                pointers.len(),
                i32::from(operation == ReduceOperation::Mean),
                outputs.as_mut_ptr(),
            )
        };
        if result != 0 {
            return Err(ffi_error());
        }
        let outputs = outputs
            .into_iter()
            .map(|pointer| {
                assert!(
                    !pointer.is_null(),
                    "native adapter must return every completed tensor"
                );
                TchTensor::new(unsafe { tch::Tensor::from_ptr(pointer.cast()) })
            })
            .collect::<Vec<_>>();
        if outputs.iter().zip(inputs).any(|(output, input)| {
            output.shape() != input.tensor.shape() || output.dtype() != input.tensor.dtype()
        }) {
            return Err(DeviceError(
                "native adapter changed collective tensor shape/dtype".into(),
            ));
        }
        Ok(outputs)
    }
}

static COMMUNICATORS: OnceLock<NativeCommunicators<TorchDevice>> = OnceLock::new();

impl<E: TchElement> DistributedBackend for LibTorch<E> {
    fn all_reduce(
        tensor: TchTensor,
        operation: ReduceOperation,
        devices: Vec<DeviceId>,
    ) -> CollectiveTensor<Self> {
        let tch::Device::Cuda(index) = tensor.tensor.device() else {
            panic!("ruCCL LibTorch native collectives require CUDA tensors");
        };
        let rank = devices
            .iter()
            .position(|id| id.type_id == 0 && id.index_id as usize == index)
            .expect("tensor device must belong to the ruCCL group");
        if devices.len() == 1 {
            return CollectiveTensor::new(tensor);
        }
        let communicator = COMMUNICATORS
            .get_or_init(|| NativeCommunicators::new(TorchDevice))
            .communicator(devices)
            .expect("valid ruCCL native communicator");
        let output = communicator
            .all_reduce(rank, tensor, operation)
            .unwrap_or_else(|error| panic!("ruCCL LibTorch all-reduce: {error}"));
        CollectiveTensor::new(output)
    }

    fn sync_collective(_device: &LibTorchDevice) {
        // ruCCL returns only after the device executor completes the captured streams.
    }
}
