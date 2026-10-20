use alloc::string::String;
use bytemuck::checked::CheckedCastError;
use thiserror::Error;
use crate::bytes::AccessError;
use crate::tensor::{DType, execution::ExecutionError};

/// The things that can go wrong when manipulating tensor data.
#[derive(Debug, Error)]
pub enum DataError {
    /// Host storage could not be read or written.
    #[error("Failed to access TensorData storage: {0}")]
    StorageAccess(#[from] AccessError),
    /// The bytes are invalid for the requested element type.
    #[error("TensorData storage is invalid for the requested element type: {0}")]
    InvalidRepresentation(CheckedCastError),
    /// The stored dtype does not match the requested dtype.
    #[error("Expected data type {expected:?}, but got {actual:?}")]
    DTypeMismatch {
        /// Requested dtype.
        expected: DType,
        /// Stored dtype.
        actual: DType,
    },
    /// The requested conversion is not supported.
    #[error("Unsupported data conversion from {from:?} to {to:?}")]
    UnsupportedConversion {
        /// Source dtype.
        from: DType,
        /// Destination dtype.
        to: DType,
    },
    /// The shape and stored number of elements disagree.
    #[error("TensorData shape describes {expected} element(s), but storage contains {actual}")]
    ElementCountMismatch {
        /// Number of elements described by the shape.
        expected: usize,
        /// Number of stored elements.
        actual: usize,
    },
    /// Failed to cast the values to a specified element type.
    #[error("Failed to cast values to the specified element type.\nError:\n  {0}")]
    CastError(CheckedCastError),
    /// Invalid target element type.
    #[error("{0}")]
    TypeMismatch(String),
}

/// Errors reading tensor data into host memory.
#[derive(Debug, Error)]
pub enum TensorReadError {
    /// Device execution failed.
    #[error(transparent)]
    Execution(#[from] ExecutionError),
    /// Host storage access or conversion failed.
    #[error(transparent)]
    Data(#[from] DataError),
    /// The tensor does not contain the required number of elements.
    #[error("Expected {expected} tensor element(s), but got {actual}")]
    InvalidShape {
        /// Required element count.
        expected: usize,
        /// Observed element count.
        actual: usize,
    },
}
