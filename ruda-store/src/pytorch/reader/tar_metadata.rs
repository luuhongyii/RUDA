use super::{Object, PytorchError, Result, TensorSnapshot, extract_tensors_with_data, read_pickle};
use byteorder::{LittleEndian, ReadBytesExt};
use ruda_model::module::ParamId;
use ruda_tensor::api::{BoolStore, DType, TensorData};
use std::collections::HashMap;
use std::io::Cursor;
use std::rc::Rc;
use std::sync::Arc;

#[derive(Clone)]
struct Storage {
    offset: usize,
    bytes: usize,
    dtype: DType,
}

fn invalid(message: &str) -> PytorchError {
    PytorchError::InvalidFormat(message.to_owned())
}

fn integer(object: &Object) -> Result<usize> {
    match object {
        Object::Int(value) => usize::try_from(*value).map_err(|_| invalid("Invalid TAR integer")),
        _ => Err(invalid("Expected TAR integer")),
    }
}

fn identifier(object: &Object) -> Result<String> {
    match object {
        Object::String(value) => Ok(value.clone()),
        Object::Int(value) => Ok(value.to_string()),
        _ => Err(invalid("Invalid TAR object identifier")),
    }
}

fn fields(object: Object, size: usize) -> Result<Vec<Object>> {
    match object {
        Object::Tuple(values) if values.len() == size => Ok(values),
        _ => Err(invalid("Invalid TAR metadata tuple")),
    }
}

fn dtype(object: &Object) -> Result<DType> {
    let Object::Class { name, .. } = object else {
        return Err(invalid("Invalid TAR storage type"));
    };
    match name.as_str() {
        "FloatStorage" => Ok(DType::F32),
        "DoubleStorage" => Ok(DType::F64),
        "HalfStorage" => Ok(DType::F16),
        "BFloat16Storage" => Ok(DType::BF16),
        "LongStorage" => Ok(DType::I64),
        "IntStorage" => Ok(DType::I32),
        "ShortStorage" => Ok(DType::I16),
        "CharStorage" => Ok(DType::I8),
        "ByteStorage" => Ok(DType::U8),
        "BoolStorage" => Ok(DType::Bool(BoolStore::Native)),
        _ => Err(invalid("Unsupported TAR storage type")),
    }
}

fn product(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("TAR metadata overflow"))
}

fn sum(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| invalid("TAR metadata overflow"))
}

fn storages(data: &[u8]) -> Result<HashMap<String, Storage>> {
    let mut cursor = Cursor::new(data);
    let count = integer(&read_pickle(&mut cursor)?)?;
    let mut result = HashMap::new();
    for _ in 0..count {
        let meta = fields(read_pickle(&mut cursor)?, 3)?;
        let dtype = dtype(&meta[2])?;
        let count = usize::try_from(cursor.read_u64::<LittleEndian>()?)
            .map_err(|_| invalid("Invalid TAR storage length"))?;
        let offset = cursor.position() as usize;
        let bytes = product(count, dtype.size())?;
        let end = sum(offset, bytes)?;
        if end > data.len() {
            return Err(invalid("Truncated TAR storage"));
        }
        result.insert(
            identifier(&meta[0])?,
            Storage {
                offset,
                bytes,
                dtype,
            },
        );
        cursor.set_position(end as u64);
    }
    if cursor.position() < data.len() as u64 {
        let Object::List(views) = read_pickle(&mut cursor)? else {
            return Err(invalid("Invalid TAR storage views"));
        };
        for view in views {
            let meta = fields(view, 4)?;
            let root = result
                .get(&identifier(&meta[1])?)
                .ok_or_else(|| invalid("TAR storage view root not found"))?
                .clone();
            let offset = product(integer(&meta[2])?, root.dtype.size())?;
            let bytes = product(integer(&meta[3])?, root.dtype.size())?;
            if sum(offset, bytes)? > root.bytes {
                return Err(invalid("TAR storage view exceeds root storage"));
            }
            result.insert(
                identifier(&meta[0])?,
                Storage {
                    offset: sum(root.offset, offset)?,
                    bytes,
                    dtype: root.dtype,
                },
            );
        }
    }
    Ok(result)
}

fn tensors(data: &[u8], storage_data: Arc<Vec<u8>>) -> Result<HashMap<String, TensorSnapshot>> {
    let storages = storages(&storage_data)?;
    let mut cursor = Cursor::new(data);
    let count = integer(&read_pickle(&mut cursor)?)?;
    let mut result = HashMap::new();
    for _ in 0..count {
        let meta = fields(read_pickle(&mut cursor)?, 3)?;
        let storage = storages
            .get(&identifier(&meta[1])?)
            .ok_or_else(|| invalid("TAR tensor storage not found"))?
            .clone();
        let rank = usize::try_from(cursor.read_i32::<LittleEndian>()?)
            .map_err(|_| invalid("Invalid TAR tensor rank"))?;
        cursor.read_u32::<LittleEndian>()?;
        if rank > (data.len().saturating_sub(cursor.position() as usize)) / 16 {
            return Err(invalid("Truncated TAR tensor dimensions"));
        }
        let mut dimensions = |count| -> Result<Vec<usize>> {
            (0..count)
                .map(|_| {
                    usize::try_from(cursor.read_i64::<LittleEndian>()?)
                        .map_err(|_| invalid("Negative TAR tensor dimension, stride or offset"))
                })
                .collect()
        };
        let shape = dimensions(rank)?;
        let strides = dimensions(rank)?;
        let offset = dimensions(1)?[0];
        let count = shape.iter().try_fold(1usize, |a, b| product(a, *b))?;
        let element_size = storage.dtype.size();
        let bytes = product(count, element_size)?;
        if count > 0 {
            let last = shape
                .iter()
                .zip(&strides)
                .try_fold(offset, |acc, (dim, stride)| {
                    sum(acc, product(dim - 1, *stride)?)
                })?;
            if product(sum(last, 1)?, element_size)? > storage.bytes {
                return Err(invalid("TAR tensor exceeds storage bounds"));
            }
        }
        let mut contiguous = true;
        let mut expected_stride = 1;
        for (&dim, &stride) in shape.iter().zip(&strides).rev() {
            contiguous &= dim <= 1 || stride == expected_stride;
            expected_stride = product(expected_stride, dim)?;
        }
        let output_shape = shape.clone();
        let blob = storage_data.clone();
        let data_fn = Rc::new(move || {
            let mut values = Vec::with_capacity(bytes);
            if count > 0 && contiguous {
                let start = storage.offset + offset * element_size;
                values.extend_from_slice(&blob[start..start + bytes]);
            } else {
                for flat in 0..count {
                    let mut remaining = flat;
                    let mut index = offset;
                    for (&dim, &stride) in shape.iter().zip(&strides).rev() {
                        index += (remaining % dim) * stride;
                        remaining /= dim;
                    }
                    let start = storage.offset + index * element_size;
                    values.extend_from_slice(&blob[start..start + element_size]);
                }
            }
            if cfg!(target_endian = "big") {
                for element in values.chunks_exact_mut(element_size) {
                    element.reverse();
                }
            }
            Ok(TensorData::from_bytes_vec(
                values,
                shape.clone(),
                storage.dtype,
            ))
        });
        result.insert(
            identifier(&meta[0])?,
            TensorSnapshot::from_closure(
                data_fn,
                storage.dtype,
                output_shape.into(),
                vec![],
                vec![],
                ParamId::new(),
            ),
        );
    }
    Ok(result)
}

fn resolve(object: Object, tensors: &HashMap<String, TensorSnapshot>) -> Result<Object> {
    match object {
        Object::Persistent(id) => {
            let key = String::from_utf8(id).map_err(|_| invalid("Invalid TAR tensor reference"))?;
            let tensor = tensors
                .get(&key)
                .ok_or_else(|| invalid("TAR tensor reference not found"))?;
            Ok(Object::TorchParam(tensor.clone()))
        }
        Object::Dict(values) => Ok(Object::Dict(
            values
                .into_iter()
                .map(|(key, value)| Ok((key, resolve(value, tensors)?)))
                .collect::<Result<_>>()?,
        )),
        Object::Build { callable, args } => {
            let parameter = matches!(callable.as_ref(), Object::Reduce { callable, .. }
                if matches!(callable.as_ref(), Object::Class { module_name, name }
                    if module_name == "torch.nn.parameter" && name == "Parameter"));
            if parameter {
                let Object::Tuple(mut state) = *args else {
                    return Err(invalid("Invalid TAR Parameter state"));
                };
                if state.is_empty() {
                    return Err(invalid("Empty TAR Parameter state"));
                }
                resolve(state.remove(0), tensors)
            } else {
                Ok(Object::Build { callable, args })
            }
        }
        other => Ok(other),
    }
}

pub(super) fn read(
    pickle_data: &[u8],
    tensor_data: &[u8],
    storage_data: Vec<u8>,
    top_level_key: Option<&str>,
) -> Result<HashMap<String, TensorSnapshot>> {
    let tensors = tensors(tensor_data, Arc::new(storage_data))?;
    let object = read_pickle(&mut Cursor::new(pickle_data))?;
    extract_tensors_with_data(resolve(object, &tensors)?, top_level_key)
}
