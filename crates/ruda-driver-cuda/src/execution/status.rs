use cudarc::driver::{DriverError, sys::CUresult};
use ruda_core::{backtrace::BackTrace, compiler::CompilationError, poison::DevicePoison};
use ruda::runtime::server::{IoError, LaunchError, ServerError};

pub(crate) fn poisons_device(status: CUresult) -> bool {
    matches!(status,
        CUresult::CUDA_ERROR_ECC_UNCORRECTABLE
        | CUresult::CUDA_ERROR_ILLEGAL_ADDRESS
        | CUresult::CUDA_ERROR_LAUNCH_TIMEOUT
        | CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED
        | CUresult::CUDA_ERROR_ASSERT
        | CUresult::CUDA_ERROR_HARDWARE_STACK_ERROR
        | CUresult::CUDA_ERROR_ILLEGAL_INSTRUCTION
        | CUresult::CUDA_ERROR_MISALIGNED_ADDRESS
        | CUresult::CUDA_ERROR_INVALID_ADDRESS_SPACE
        | CUresult::CUDA_ERROR_INVALID_PC
        | CUresult::CUDA_ERROR_LAUNCH_FAILED)
}

pub(crate) fn server(op: &str, error: DriverError) -> ServerError {
    let reason = format!("{op}: {error:?}");
    if poisons_device(error.0) { DevicePoison::new(reason).into() }
    else { ServerError::Generic { reason, backtrace: BackTrace::capture() } }
}

pub(crate) fn checked(op: &str, status: CUresult) -> Result<(), ServerError> {
    if status == CUresult::CUDA_SUCCESS { Ok(()) }
    else { Err(server(op, DriverError(status))) }
}

pub(crate) fn io(op: &str, error: DriverError) -> IoError {
    let reason = format!("{op}: {error:?}");
    if poisons_device(error.0) { DevicePoison::new(reason).into() }
    else { IoError::Unknown { description: reason, backtrace: BackTrace::capture() } }
}

pub(crate) fn compilation(op: &str, error: DriverError) -> CompilationError {
    let reason = format!("{op}: {error:?}");
    if poisons_device(error.0) { DevicePoison::new(reason).into() }
    else { CompilationError::Generic { reason, backtrace: BackTrace::capture() } }
}

pub(crate) fn launch(op: &str, error: DriverError) -> LaunchError {
    let reason = format!("{op}: {error:?}");
    if poisons_device(error.0) { DevicePoison::new(reason).into() }
    else if error.0 == CUresult::CUDA_ERROR_OUT_OF_MEMORY {
        LaunchError::OutOfMemory { reason, backtrace: BackTrace::capture() }
    } else { LaunchError::Unknown { reason, backtrace: BackTrace::capture() } }
}
