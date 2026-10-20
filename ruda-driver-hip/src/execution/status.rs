use ruda_hip_sys::*;
use ruda_core::{backtrace::BackTrace, compiler::CompilationError, poison::DevicePoison};
use ruda::runtime::server::{IoError, LaunchError, ServerError};

#[allow(non_upper_case_globals)]
pub(crate) fn poisons_device(status: hipError_t) -> bool {
    matches!(status, hipError_t_hipErrorECCNotCorrectable
        | hipError_t_hipErrorIllegalAddress
        | hipError_t_hipErrorLaunchTimeOut
        | hipError_t_hipErrorContextIsDestroyed
        | hipError_t_hipErrorAssert
        | hipError_t_hipErrorLaunchFailure)
}

pub(crate) fn server(op: &str, status: hipError_t) -> ServerError {
    let reason = format!("{op}: HIP status {status}");
    if poisons_device(status) { DevicePoison::new(reason).into() }
    else { ServerError::Generic { reason, backtrace: BackTrace::capture() } }
}

pub(crate) fn checked(op: &str, status: hipError_t) -> Result<(), ServerError> {
    if status == HIP_SUCCESS { Ok(()) } else { Err(server(op, status)) }
}

pub(crate) fn io(op: &str, status: hipError_t) -> IoError {
    let reason = format!("{op}: HIP status {status}");
    if poisons_device(status) { DevicePoison::new(reason).into() }
    else { IoError::Unknown { description: reason, backtrace: BackTrace::capture() } }
}

pub(crate) fn compilation(op: &str, status: hipError_t) -> CompilationError {
    let reason = format!("{op}: HIP status {status}");
    if poisons_device(status) { DevicePoison::new(reason).into() }
    else { CompilationError::Generic { reason, backtrace: BackTrace::capture() } }
}

pub(crate) fn launch(op: &str, status: hipError_t) -> LaunchError {
    let reason = format!("{op}: HIP status {status}");
    if poisons_device(status) { DevicePoison::new(reason).into() }
    else if status == hipError_t_hipErrorOutOfMemory {
        LaunchError::OutOfMemory { reason, backtrace: BackTrace::capture() }
    } else { LaunchError::Unknown { reason, backtrace: BackTrace::capture() } }
}
