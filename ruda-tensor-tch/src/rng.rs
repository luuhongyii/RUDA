use crate::LibTorchDevice;
use ruda_tensor::backend::ExecutionError;
use std::{ffi::{CStr, c_char}, ptr::NonNull};

unsafe extern "C" {
    fn ruda_tch_rng_get(kind: i32, index: i32, length: *mut usize) -> *mut u8;
    fn ruda_tch_rng_free(bytes: *mut u8);
    fn ruda_tch_rng_set(kind: i32, index: i32, bytes: *const u8, length: usize) -> i32;
    fn ruda_tch_rng_error() -> *const c_char;
}

fn error(reason: impl Into<String>) -> ExecutionError {
    ExecutionError::WithContext { reason: reason.into() }
}

fn ffi_error() -> ExecutionError {
    error(unsafe { CStr::from_ptr(ruda_tch_rng_error()) }.to_string_lossy().into_owned())
}

fn device_args(device: &LibTorchDevice) -> Result<(i32, i32), ExecutionError> {
    Ok(match device {
        LibTorchDevice::Cpu => (0, 0),
        LibTorchDevice::Cuda(index) => (1, i32::try_from(*index)
            .map_err(|_| error("LibTorch RNG device index exceeds i32"))?),
        LibTorchDevice::Mps => (2, 0),
        LibTorchDevice::Vulkan => return Err(error("LibTorch Vulkan has no default RNG generator")),
    })
}

struct StateBytes(NonNull<u8>);
impl Drop for StateBytes {
    fn drop(&mut self) { unsafe { ruda_tch_rng_free(self.0.as_ptr()) }; }
}

pub(crate) fn capture(device: &LibTorchDevice) -> Result<Vec<u8>, ExecutionError> {
    let (kind, index) = device_args(device)?;
    let mut length = 0;
    let bytes = StateBytes(NonNull::new(unsafe { ruda_tch_rng_get(kind, index, &mut length) })
        .ok_or_else(ffi_error)?);
    let mut state = Vec::with_capacity(length.checked_add(5)
        .ok_or_else(|| error("LibTorch RNG checkpoint size overflow"))?);
    state.extend_from_slice(b"TR01");
    state.push(kind as u8);
    state.extend_from_slice(unsafe { std::slice::from_raw_parts(bytes.0.as_ptr(), length) });
    Ok(state)
}

pub(crate) fn restore(device: &LibTorchDevice, state: &[u8]) -> Result<(), ExecutionError> {
    let (kind, index) = device_args(device)?;
    if state.len() <= 5 || &state[..4] != b"TR01" || state[4] != kind as u8 {
        return Err(error("LibTorch RNG checkpoint version or device kind differs"));
    }
    let bytes = &state[5..];
    if unsafe { ruda_tch_rng_set(kind, index, bytes.as_ptr(), bytes.len()) } != 0 {
        return Err(ffi_error());
    }
    Ok(())
}
