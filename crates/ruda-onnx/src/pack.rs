//! Deferred ONNX initializer conversion into RUDA's native model store.

use std::{fmt, path::Path, rc::Rc};
use burn_tensor::{DType, Shape, TensorData};
use ruda_model::module::ParamId;
use ruda_store::{RudapackWriter, TensorSnapshot, TensorSnapshotError};
use ruda_tensor::api as runtime;

pub type Tensor = TensorSnapshot;

#[derive(Debug)]
pub enum Error {
    ValidationError(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Self::ValidationError(message) => f.write_str(message) }
    }
}
impl std::error::Error for Error {}

pub(crate) fn dtype(value: DType) -> runtime::DType {
    match value {
        DType::F16 => runtime::DType::F16,
        DType::BF16 => runtime::DType::BF16,
        DType::F32 => runtime::DType::F32,
        DType::F64 => runtime::DType::F64,
        DType::I8 => runtime::DType::I8,
        DType::I16 => runtime::DType::I16,
        DType::I32 => runtime::DType::I32,
        DType::I64 => runtime::DType::I64,
        DType::U8 => runtime::DType::U8,
        DType::U16 => runtime::DType::U16,
        DType::U32 => runtime::DType::U32,
        DType::U64 => runtime::DType::U64,
        DType::Bool(store) => runtime::DType::Bool(match store {
            burn_tensor::BoolStore::Native => runtime::BoolStore::Native,
            burn_tensor::BoolStore::U8 => runtime::BoolStore::U8,
            burn_tensor::BoolStore::U32 => runtime::BoolStore::U32,
        }),
        other => panic!("ONNX initializer has unsupported dtype {other:?}"),
    }
}

pub(crate) fn data(value: TensorData) -> runtime::TensorData {
    runtime::TensorData::from_bytes_vec(
        value.as_bytes().to_vec(), value.shape().to_vec(), dtype(value.dtype()),
    )
}

pub mod bridge {
    use super::*;

    pub fn deferred(
        path: String,
        tensor_dtype: DType,
        shape: Shape,
        id: Option<u64>,
        provider: impl Fn() -> Result<TensorData, Error> + 'static,
    ) -> Tensor {
        let error_path = path.clone();
        TensorSnapshot::from_closure(
            Rc::new(move || provider().map(data).map_err(|error| {
                TensorSnapshotError::DataError(format!("{error_path}: {error}"))
            })),
            dtype(tensor_dtype),
            runtime::Shape::from(shape.to_vec()),
            path.split('.').map(str::to_owned).collect(),
            Vec::new(),
            id.map(ParamId::from).unwrap_or_default(),
        )
    }

    pub fn from_data(value: TensorData, path: String, id: Option<u64>) -> Tensor {
        from_ruda_data(data(value), path, id)
    }

    pub fn from_ruda_data(value: runtime::TensorData, path: String, id: Option<u64>) -> Tensor {
        TensorSnapshot::from_data(value, path.split('.').map(str::to_owned).collect(),
                                  Vec::new(), id.map(ParamId::from).unwrap_or_default())
    }
}

pub struct Writer(RudapackWriter);

impl Writer {
    pub fn new(tensors: Vec<Tensor>) -> Self { Self(RudapackWriter::new(tensors)) }

    pub fn with_metadata(self, key: &str, value: &str) -> Self {
        Self(self.0.with_metadata(key, value))
    }

    pub fn into_bytes(self) -> Result<runtime::Bytes, Error> {
        self.0.to_bytes().map_err(|e| Error::ValidationError(e.to_string()))
    }

    pub fn write_to_file(self, path: impl AsRef<Path>) -> Result<(), Error> {
        let path = path.as_ref();
        let parent = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
        let temporary = tempfile::NamedTempFile::new_in(parent)
            .map_err(|e| Error::ValidationError(e.to_string()))?;
        self.0.write_to_file(temporary.path())
            .map_err(|e| Error::ValidationError(e.to_string()))?;
        temporary.persist(path).map_err(|e| Error::ValidationError(e.to_string()))?;
        Ok(())
    }
}
