use super::*;
use ruda_tensor::api::TensorData;
use std::{fs::File, io::{BufReader, Read, Seek, SeekFrom}, ops::Range};

pub(crate) struct StackedMatrixSource {
    file: BufReader<File>,
    start: u64,
    shape: [usize; 3],
    dtype: DType,
}

impl StackedMatrixSource {
    pub(crate) fn matrix(&mut self, expert: usize, rows: Range<usize>, columns: Range<usize>)
        -> Result<TensorData, HuggingFaceLoadError> {
        let [experts, height, width] = self.shape;
        if expert >= experts || rows.is_empty() || columns.is_empty() || rows.end > height || columns.end > width {
            return Err(HuggingFaceLoadError("invalid stacked expert matrix selection".into()));
        }
        let size = self.dtype.size();
        let count = rows.len().checked_mul(columns.len()).and_then(|n| n.checked_mul(size))
            .ok_or_else(|| HuggingFaceLoadError("stacked expert shard byte count overflow".into()))?;
        let mut bytes = vec![0; count];
        if columns.start == 0 && columns.end == width {
            let offset = ((expert * height + rows.start) * width * size) as u64;
            self.file.seek(SeekFrom::Start(self.start + offset)).map_err(io_error)?;
            self.file.read_exact(&mut bytes).map_err(io_error)?;
        } else {
            let stride = columns.len() * size;
            let skip = i64::try_from((width - columns.len()) * size)
                .map_err(|_| HuggingFaceLoadError("stacked matrix row stride exceeds file indexing".into()))?;
            let offset = (((expert * height + rows.start) * width + columns.start) * size) as u64;
            self.file.seek(SeekFrom::Start(self.start + offset)).map_err(io_error)?;
            for local in 0..rows.len() {
                self.file.read_exact(&mut bytes[local * stride..(local + 1) * stride]).map_err(io_error)?;
                if local + 1 < rows.len() { self.file.seek_relative(skip).map_err(io_error)?; }
            }
        }
        Ok(TensorData::from_bytes_vec(bytes, [rows.len(), columns.len()], self.dtype))
    }
}

impl Checkpoint {
    pub(crate) fn stacked_matrix_source(&mut self, name: &str, shape: [usize; 3])
        -> Result<StackedMatrixSource, HuggingFaceLoadError> {
        let snapshot = self.tensors.get(name).ok_or_else(|| HuggingFaceLoadError(format!("missing tensor {name}")))?;
        if snapshot.shape != shape.into() || !matches!(snapshot.dtype, DType::F16 | DType::BF16 | DType::F32) {
            return Err(HuggingFaceLoadError(format!("{name}: expected a floating stacked matrix {shape:?}")));
        }
        let dtype = snapshot.dtype;
        let count = shape.iter().try_fold(dtype.size(), |n, &d| n.checked_mul(d))
            .ok_or_else(|| HuggingFaceLoadError(format!("{name}: stacked tensor byte count overflow")))?;
        for path in &self.files {
            let mut file = File::open(path).map_err(io_error)?;
            let file_length = file.metadata().map_err(io_error)?.len();
            let mut prefix = [0; 8];
            file.read_exact(&mut prefix).map_err(io_error)?;
            let header_length = u64::from_le_bytes(prefix);
            let payload = header_length.checked_add(8).filter(|&n| n <= file_length)
                .ok_or_else(|| HuggingFaceLoadError("invalid stacked safetensors header length".into()))?;
            let header: BTreeMap<String, serde_json::Value> = serde_json::from_reader((&mut file).take(header_length))
                .map_err(|e| HuggingFaceLoadError(e.to_string()))?;
            let Some(entry) = header.get(name) else { continue; };
            let offsets: [u64; 2] = serde_json::from_value(entry["data_offsets"].clone())
                .map_err(|e| HuggingFaceLoadError(e.to_string()))?;
            let stored_shape: [usize; 3] = serde_json::from_value(entry["shape"].clone())
                .map_err(|e| HuggingFaceLoadError(e.to_string()))?;
            let expected_dtype = match dtype { DType::F16 => "F16", DType::BF16 => "BF16", _ => "F32" };
            if stored_shape != shape || entry["dtype"] != expected_dtype || offsets[0] > offsets[1]
                || offsets[1] > file_length - payload || offsets[1] - offsets[0] != count as u64 {
                return Err(HuggingFaceLoadError(format!("{name}: stacked safetensors extent differs from tensor metadata")));
            }
            self.consumed.insert(name.into());
            return Ok(StackedMatrixSource { file: BufReader::with_capacity(128 * 1024, file), start: payload + offsets[0], shape, dtype });
        }
        Err(HuggingFaceLoadError(format!("{name}: source safetensors extent not found")))
    }
}
