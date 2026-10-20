use std::{
    collections::BTreeMap,
    fs::File,
    io::{self, BufReader, Read, Seek, SeekFrom},
    ops::Range,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
pub use ruda_core::tensor::quantization::GgmlType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgufEndian { Little, Big }

#[derive(Clone, Debug, PartialEq)]
pub enum GgufValue {
    U8(u8), I8(i8), U16(u16), I16(i16), U32(u32), I32(i32),
    F32(f32), Bool(bool), String(String), Array { element_type: u32, values: Vec<Self> },
    U64(u64), I64(i64), F64(f64),
}

impl GgufValue {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U8(v) => Some(*v as u64), Self::U16(v) => Some(*v as u64),
            Self::U32(v) => Some(*v as u64), Self::U64(v) => Some(*v),
            Self::I8(v) => u64::try_from(*v).ok(), Self::I16(v) => u64::try_from(*v).ok(),
            Self::I32(v) => u64::try_from(*v).ok(), Self::I64(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self { Self::F32(v) => Some(*v as f64), Self::F64(v) => Some(*v), _ => None }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self { Self::String(v) => Some(v), _ => None }
    }
    pub fn as_array(&self) -> Option<&[Self]> {
        match self { Self::Array { values, .. } => Some(values), _ => None }
    }
}

#[derive(Clone, Debug)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dimensions: Vec<usize>,
    pub dtype: GgmlType,
    pub offset: u64,
    pub byte_len: usize,
}

impl GgufTensorInfo {
    pub fn shape(&self) -> Vec<usize> { self.dimensions.iter().rev().copied().collect() }
    pub fn row_bytes(&self) -> usize {
        self.dtype.row_bytes(self.dimensions[0]).expect("validated GGUF row")
    }
    pub fn rows(&self) -> usize { self.dimensions[1..].iter().product() }
}

#[derive(Clone)]
pub struct GgufFile {
    pub path: PathBuf,
    pub version: u32,
    pub endian: GgufEndian,
    pub metadata: BTreeMap<String, GgufValue>,
    pub tensors: BTreeMap<String, GgufTensorInfo>,
    data_offset: u64,
    file: Arc<Mutex<File>>,
}

#[derive(Clone, Debug)]
pub struct GgufPackedTensor {
    pub info: GgufTensorInfo,
    pub endian: GgufEndian,
    pub bytes: Vec<u8>,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn unsigned_metadata(metadata: &BTreeMap<String, GgufValue>, key: &str) -> io::Result<Option<u64>> {
    metadata.get(key).map(|value| value.as_u64()
        .ok_or_else(|| invalid(format!("GGUF {key} must be a nonnegative integer")))).transpose()
}

struct Reader {
    file: BufReader<File>,
    endian: GgufEndian,
    len: u64,
}

impl Reader {
    fn bytes<const N: usize>(&mut self) -> io::Result<[u8; N]> {
        let mut bytes = [0; N];
        self.file.read_exact(&mut bytes)?;
        Ok(bytes)
    }
    fn u8(&mut self) -> io::Result<u8> { Ok(self.bytes::<1>()?[0]) }
    fn u16(&mut self) -> io::Result<u16> {
        let bytes = self.bytes()?;
        Ok(match self.endian { GgufEndian::Little => u16::from_le_bytes(bytes), GgufEndian::Big => u16::from_be_bytes(bytes) })
    }
    fn u32(&mut self) -> io::Result<u32> {
        let bytes = self.bytes()?;
        Ok(match self.endian { GgufEndian::Little => u32::from_le_bytes(bytes), GgufEndian::Big => u32::from_be_bytes(bytes) })
    }
    fn u64(&mut self) -> io::Result<u64> {
        let bytes = self.bytes()?;
        Ok(match self.endian { GgufEndian::Little => u64::from_le_bytes(bytes), GgufEndian::Big => u64::from_be_bytes(bytes) })
    }
    fn count(&mut self, count: u64, minimum_bytes: u64) -> io::Result<usize> {
        let remaining = self.len.saturating_sub(self.file.stream_position()?);
        if count > remaining / minimum_bytes {
            return Err(invalid("GGUF count exceeds remaining file data"));
        }
        usize::try_from(count).map_err(|_| invalid("GGUF count exceeds address space"))
    }
    fn string(&mut self) -> io::Result<String> {
        let count = self.u64()?;
        let count = self.count(count, 1)?;
        let mut bytes = vec![0; count];
        self.file.read_exact(&mut bytes)?;
        String::from_utf8(bytes).map_err(|e| invalid(format!("GGUF string: {e}")))
    }
    fn value(&mut self, ty: u32) -> io::Result<GgufValue> {
        Ok(match ty {
            0 => GgufValue::U8(self.u8()?),
            1 => GgufValue::I8(self.u8()? as i8),
            2 => GgufValue::U16(self.u16()?),
            3 => GgufValue::I16(self.u16()? as i16),
            4 => GgufValue::U32(self.u32()?),
            5 => GgufValue::I32(self.u32()? as i32),
            6 => GgufValue::F32(f32::from_bits(self.u32()?)),
            7 => GgufValue::Bool(match self.u8()? {
                0 => false, 1 => true, _ => return Err(invalid("invalid GGUF boolean")),
            }),
            8 => GgufValue::String(self.string()?),
            9 => {
                let element_type = self.u32()?;
                let minimum_bytes = match element_type {
                    0 | 1 | 7 => 1, 2 | 3 => 2, 4 | 5 | 6 => 4, 8 | 10 | 11 | 12 => 8, 9 => 12,
                    _ => return Err(invalid("invalid GGUF array element type")),
                };
                let count = self.u64()?;
                let count = self.count(count, minimum_bytes)?;
                let mut values = Vec::with_capacity(count);
                for _ in 0..count { values.push(self.value(element_type)?); }
                GgufValue::Array { element_type, values }
            }
            10 => GgufValue::U64(self.u64()?),
            11 => GgufValue::I64(self.u64()? as i64),
            12 => GgufValue::F64(f64::from_bits(self.u64()?)),
            _ => return Err(invalid(format!("unknown GGUF metadata type {ty}"))),
        })
    }
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let len = file.metadata()?.len();
        let mut reader = Reader { file: BufReader::new(file), endian: GgufEndian::Little, len };
        if reader.bytes::<4>()? != *b"GGUF" { return Err(invalid("not a GGUF file")); }
        let version = reader.bytes::<4>()?;
        let version = match u32::from_le_bytes(version) {
            v @ (2 | 3) => v,
            _ => {
                reader.endian = GgufEndian::Big;
                let v = u32::from_be_bytes(version);
                if v != 3 { return Err(invalid("unsupported GGUF version")); }
                v
            }
        };
        let tensors_count = reader.u64()?;
        let metadata_count = reader.u64()?;
        let metadata_count = reader.count(metadata_count, 12)?;
        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_count {
            let key = reader.string()?;
            if key.len() > u16::MAX as usize || !key.is_ascii() {
                return Err(invalid("GGUF metadata keys must be ASCII and at most 65535 bytes"));
            }
            let ty = reader.u32()?;
            if metadata.insert(key, reader.value(ty)?).is_some() {
                return Err(invalid("duplicate GGUF metadata key"));
            }
        }
        let alignment = match metadata.get("general.alignment") {
            Some(GgufValue::U32(value)) if value.is_power_of_two() => u64::from(*value),
            None => 32,
            _ => return Err(invalid("GGUF alignment must be a nonzero power-of-two uint32")),
        };
        let tensors_count = reader.count(tensors_count, 32)?;
        let mut tensors = BTreeMap::new();
        for _ in 0..tensors_count {
            let name = reader.string()?;
            if name.len() > 64 { return Err(invalid("GGUF tensor name exceeds 64 bytes")); }
            let rank = reader.u32()?;
            if !(1..=4).contains(&rank) { return Err(invalid("GGUF tensor rank must be 1..=4")); }
            let mut dimensions = Vec::with_capacity(rank as usize);
            for _ in 0..rank {
                let dim = usize::try_from(reader.u64()?).map_err(|_| invalid("GGUF dimension overflow"))?;
                if dim == 0 { return Err(invalid("empty GGUF tensor dimension")); }
                dimensions.push(dim);
            }
            let id = reader.u32()?;
            let dtype = GgmlType::from_id(id).ok_or_else(|| invalid(format!("unsupported GGML type {id}")))?;
            let row_bytes = dtype.row_bytes(dimensions[0]).ok_or_else(|| invalid("GGUF row is not block aligned"))?;
            let byte_len = dimensions[1..].iter().try_fold(row_bytes, |bytes, dim| bytes.checked_mul(*dim))
                .ok_or_else(|| invalid("GGUF tensor size overflow"))?;
            let offset = reader.u64()?;
            if offset % alignment != 0 { return Err(invalid("unaligned GGUF tensor offset")); }
            let info = GgufTensorInfo { name: name.clone(), dimensions, dtype, offset, byte_len };
            if tensors.insert(name, info).is_some() { return Err(invalid("duplicate GGUF tensor name")); }
        }
        let position = reader.file.stream_position()?;
        let data_offset = position.checked_add(alignment - 1).ok_or_else(|| invalid("GGUF offset overflow"))?
            & !(alignment - 1);
        let mut intervals = Vec::with_capacity(tensors.len());
        for info in tensors.values() {
            let start = data_offset.checked_add(info.offset).ok_or_else(|| invalid("GGUF offset overflow"))?;
            let end = start.checked_add(info.byte_len as u64).ok_or_else(|| invalid("GGUF size overflow"))?;
            if end > len { return Err(invalid(format!("truncated GGUF tensor {}", info.name))); }
            intervals.push((start, end));
        }
        intervals.sort_unstable();
        if intervals.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err(invalid("overlapping GGUF tensor ranges"));
        }
        Ok(Self { path, version, endian: reader.endian, metadata, tensors, data_offset,
            file: Arc::new(Mutex::new(reader.file.into_inner())) })
    }

    pub fn architecture(&self) -> io::Result<&str> {
        self.metadata.get("general.architecture").and_then(GgufValue::as_str)
            .ok_or_else(|| invalid("missing GGUF general.architecture"))
    }

    pub fn read_packed(&self, name: &str) -> io::Result<GgufPackedTensor> {
        let info = self.tensors.get(name).ok_or_else(|| invalid(format!("missing GGUF tensor {name}")))?;
        let bytes = self.read_packed_rows(name, 0..info.rows())?;
        Ok(GgufPackedTensor { info: info.clone(), endian: self.endian, bytes })
    }

    pub fn read_packed_rows(&self, name: &str, rows: Range<usize>) -> io::Result<Vec<u8>> {
        let info = self.tensors.get(name).ok_or_else(|| invalid(format!("missing GGUF tensor {name}")))?;
        if rows.start > rows.end || rows.end > info.rows() { return Err(invalid("GGUF row range out of bounds")); }
        let row_bytes = info.row_bytes();
        let mut bytes = vec![0; rows.len() * row_bytes];
        let mut file = self.file.lock().map_err(|_| io::Error::other("GGUF reader lock poisoned"))?;
        file.seek(SeekFrom::Start(self.data_offset + info.offset + (rows.start * row_bytes) as u64))?;
        file.read_exact(&mut bytes)?;
        Ok(bytes)
    }
}

pub struct GgufModel {
    pub files: Vec<GgufFile>,
    tensors: BTreeMap<String, usize>,
}

impl GgufModel {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let first = GgufFile::open(path.as_ref())?;
        let count = unsigned_metadata(&first.metadata, "split.count")?.unwrap_or(1);
        if count == 1 { return Self::from_files(vec![first]); }
        let count = usize::try_from(count).map_err(|_| invalid("GGUF shard count overflow"))?;
        let name = path.as_ref().file_name().and_then(|v| v.to_str()).ok_or_else(|| invalid("invalid GGUF shard filename"))?;
        let suffix = format!("-of-{count:05}.gguf");
        let stem = name.strip_suffix(&suffix).ok_or_else(|| invalid("GGUF split filename does not match shard count"))?;
        let (base, _) = stem.rsplit_once('-').ok_or_else(|| invalid("invalid GGUF split filename"))?;
        let mut files = Vec::with_capacity(count);
        for index in 1..=count {
            let shard = path.as_ref().with_file_name(format!("{base}-{index:05}-of-{count:05}.gguf"));
            files.push(if shard == first.path { first.clone() } else { GgufFile::open(shard)? });
        }
        Self::from_files(files)
    }

    pub fn from_files(mut files: Vec<GgufFile>) -> io::Result<Self> {
        if files.is_empty() { return Err(invalid("no GGUF files")); }
        for file in &files {
            for key in ["split.no", "split.count", "split.tensors.count"] {
                unsigned_metadata(&file.metadata, key)?;
            }
        }
        files.sort_by_key(|file| file.metadata.get("split.no").and_then(GgufValue::as_u64).unwrap_or(0));
        files[0].architecture()?;
        let mut tensors = BTreeMap::new();
        for (index, file) in files.iter().enumerate() {
            let count = unsigned_metadata(&file.metadata, "split.count")?.unwrap_or(1);
            let number = unsigned_metadata(&file.metadata, "split.no")?.unwrap_or(0);
            if count != files.len() as u64 || number != index as u64 {
                return Err(invalid("inconsistent GGUF shard metadata"));
            }
            for (key, value) in &file.metadata {
                if key != "split.no" && key != "general.alignment"
                    && files[0].metadata.get(key).is_some_and(|first| first != value)
                { return Err(invalid(format!("GGUF shard metadata differs for {key}"))); }
            }
            for name in file.tensors.keys() {
                if tensors.insert(name.clone(), index).is_some() { return Err(invalid("duplicate tensor across GGUF shards")); }
            }
        }
        for file in &files {
            if let Some(count) = unsigned_metadata(&file.metadata, "split.tensors.count")? {
                if count != tensors.len() as u64 { return Err(invalid("GGUF total tensor count differs")); }
            }
        }
        Ok(Self { files, tensors })
    }

    pub fn metadata(&self) -> &BTreeMap<String, GgufValue> { &self.files[0].metadata }
    pub fn tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        let file = &self.files[*self.tensors.get(name)?];
        file.tensors.get(name)
    }
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> { self.tensors.keys().map(String::as_str) }
    pub fn read_packed(&self, name: &str) -> io::Result<GgufPackedTensor> {
        let index = self.tensors.get(name).ok_or_else(|| invalid(format!("missing GGUF tensor {name}")))?;
        self.files[*index].read_packed(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(big: bool, packed: &[u8]) -> Vec<u8> {
        fn number(out: &mut Vec<u8>, value: u64, width: usize, big: bool) {
            let bytes = if big { value.to_be_bytes() } else { value.to_le_bytes() };
            out.extend_from_slice(if big { &bytes[8 - width..] } else { &bytes[..width] });
        }
        fn string(out: &mut Vec<u8>, value: &str, big: bool) {
            number(out, value.len() as u64, 8, big);
            out.extend_from_slice(value.as_bytes());
        }
        let mut bytes = b"GGUF".to_vec();
        number(&mut bytes, 3, 4, big);
        number(&mut bytes, 1, 8, big);
        number(&mut bytes, 1, 8, big);
        string(&mut bytes, "general.architecture", big);
        number(&mut bytes, 8, 4, big);
        string(&mut bytes, "llama", big);
        string(&mut bytes, "token_embd.weight", big);
        number(&mut bytes, 2, 4, big);
        number(&mut bytes, 32, 8, big);
        number(&mut bytes, 2, 8, big);
        number(&mut bytes, 2, 4, big);
        number(&mut bytes, 0, 8, big);
        bytes.resize(bytes.len().div_ceil(32) * 32, 0);
        bytes.extend_from_slice(packed);
        bytes
    }

    #[test]
    fn native_reader_preserves_quantized_blocks_and_row_ranges_in_both_endians() {
        for big in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("model.gguf");
            let packed: Vec<u8> = (0..36).collect();
            std::fs::write(&path, fixture(big, &packed)).unwrap();
            let file = GgufFile::open(&path).unwrap();
            let tensor = file.read_packed("token_embd.weight").unwrap();
            assert_eq!(tensor.info.dtype, GgmlType::Q4_0);
            assert_eq!(tensor.info.shape(), [2, 32]);
            assert_eq!(tensor.bytes, packed);
            assert_eq!(file.read_packed_rows("token_embd.weight", 1..2).unwrap(), packed[18..]);
            assert_eq!(GgufModel::open(&path).unwrap().tensor_names().collect::<Vec<_>>(), ["token_embd.weight"]);
            drop(file);
            std::fs::write(&path, fixture(big, &packed[..35])).unwrap();
            assert!(GgufFile::open(&path).is_err());
        }
    }
}
