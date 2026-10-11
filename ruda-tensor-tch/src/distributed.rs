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
    ops::Range,
    ptr::NonNull,
    sync::OnceLock,
};

unsafe extern "C" {
    fn ruda_tch_peer_available() -> i32;
    fn ruda_tch_prepare_collective(tensor: *const c_void) -> *mut c_void;
    fn ruda_tch_drop_prepared(tensor: *mut c_void);
    fn ruda_tch_peer_begin(
        inputs: *const *const c_void,
        count: usize,
        elements: *mut usize,
    ) -> *mut c_void;
    fn ruda_tch_peer_drop(work: *mut c_void);
    fn ruda_tch_peer_exchange(
        work: *mut c_void,
        offsets: *const usize,
        lengths: *const usize,
        count: usize,
        reduce: i32,
    ) -> i32;
    fn ruda_tch_peer_finish(
        work: *mut c_void,
        count: usize,
        mean: i32,
        outputs: *mut *mut c_void,
    ) -> i32;
    fn ruda_tch_collective_error() -> *const c_char;
}

/// Whether this build supplies ruCCL with CUDA peer transfer/local arithmetic.
pub fn peer_available() -> bool {
    unsafe { ruda_tch_peer_available() != 0 }
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

struct PeerWork {
    handle: NonNull<c_void>,
    elements: usize,
    metadata: Vec<(ruda_tensor::Shape, ruda_tensor::DType)>,
}
impl Drop for PeerWork {
    fn drop(&mut self) {
        unsafe { ruda_tch_peer_drop(self.handle.as_ptr()) };
    }
}

#[derive(Clone)]
struct TorchDevice;
impl NativeCollectiveDevice for TorchDevice {
    type Tensor = TchTensor;
    type Prepared = PreparedTensor;
    type Work = PeerWork;
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

    fn begin(
        &self,
        devices: &[DeviceId],
        inputs: &[PreparedTensor],
    ) -> Result<PeerWork, DeviceError> {
        if inputs.is_empty() || inputs.len() != devices.len() {
            return Err(DeviceError(
                "ruCCL device world and prepared input count differ".into(),
            ));
        }
        let pointers = inputs
            .iter()
            .map(|input| input.handle.as_ptr().cast_const())
            .collect::<Vec<_>>();
        let mut elements = 0;
        let handle = NonNull::new(unsafe {
            ruda_tch_peer_begin(pointers.as_ptr(), pointers.len(), &mut elements)
        })
        .ok_or_else(ffi_error)?;
        Ok(PeerWork {
            handle,
            elements,
            metadata: inputs
                .iter()
                .map(|input| (input.tensor.shape(), input.tensor.dtype()))
                .collect(),
        })
    }

    fn elements(&self, work: &PeerWork) -> usize {
        work.elements
    }

    fn exchange(
        &self,
        work: &mut PeerWork,
        ranges: &[Range<usize>],
        reduce: bool,
    ) -> Result<(), DeviceError> {
        let offsets = ranges.iter().map(|range| range.start).collect::<Vec<_>>();
        let lengths = ranges.iter().map(|range| range.len()).collect::<Vec<_>>();
        if unsafe {
            ruda_tch_peer_exchange(
                work.handle.as_ptr(),
                offsets.as_ptr(),
                lengths.as_ptr(),
                ranges.len(),
                i32::from(reduce),
            )
        } != 0
        {
            return Err(ffi_error());
        }
        Ok(())
    }

    fn finish(
        &self,
        work: PeerWork,
        operation: ReduceOperation,
    ) -> Result<Vec<TchTensor>, DeviceError> {
        let mut outputs = vec![std::ptr::null_mut::<c_void>(); work.metadata.len()];
        if unsafe {
            ruda_tch_peer_finish(
                work.handle.as_ptr(),
                outputs.len(),
                i32::from(operation == ReduceOperation::Mean),
                outputs.as_mut_ptr(),
            )
        } != 0
        {
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
        if outputs
            .iter()
            .zip(&work.metadata)
            .any(|(output, (shape, dtype))| output.shape() != *shape || output.dtype() != *dtype)
        {
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
