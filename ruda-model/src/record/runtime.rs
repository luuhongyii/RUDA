use alloc::{boxed::Box, string::String, vec, vec::Vec};
use core::any::TypeId;
use ruda_tensor::{Backend, DType, TensorData, api::{Bool, Int, Tensor}};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use super::{PrecisionSettings, Record};
use crate::module::{EmptyRecord, Param, ParamId, Parameter};
use crate::module::TransformRecord;

/// Record conversion with precision selected before a module is type-erased.
pub trait RuntimeRecord<B: Backend>: Record<B> {
    /// Convert the complete record, including non-parameter fields.
    fn into_runtime(self, precision: RuntimePrecision) -> RuntimeRecordValue;
    /// Restore a record on the requested device.
    fn from_runtime(value: RuntimeRecordValue, device: &B::Device) -> Self;
}

/// Type-erased conversions for any [`PrecisionSettings`] implementation.
#[derive(Clone, Copy)]
pub struct RuntimePrecision {
    float: fn(TensorData) -> TensorData,
    int: fn(TensorData) -> TensorData,
    preserve_dtype: bool,
    pub(crate) id: TypeId,
}

impl RuntimePrecision {
    /// Retain the caller's actual floating and integer element conversions.
    pub fn new<S: PrecisionSettings>() -> Self {
        Self { float: |data| if S::PRESERVE_DTYPE { data } else { data.convert::<S::FloatElem>() },
            int: |data| if S::PRESERVE_DTYPE { data } else { data.convert::<S::IntElem>() },
            preserve_dtype: S::PRESERVE_DTYPE,
            id: if S::PRESERVE_DTYPE {
                TypeId::of::<super::NativePrecisionSettings>()
            } else {
                TypeId::of::<(S::FloatElem, S::IntElem)>()
            } }
    }

    /// Quantized tensors keep their packed representation.
    pub fn float(self, data: TensorData) -> TensorData {
        if matches!(data.dtype, DType::QFloat(_)) { data } else { (self.float)(data) }
    }

    /// Convert an integer tensor using the requested record settings.
    pub fn int(self, data: TensorData) -> TensorData { (self.int)(data) }
}

/// Structured adapter payload; scalar fields are not tensor precision conversions.
#[doc(hidden)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RuntimeRecordValue {
    Empty,
    Scalar(Vec<u8>),
    Float(TensorData),
    Int(TensorData),
    Bool(TensorData),
    Fields(u32, Vec<Self>),
    Adapter(String, Box<Self>),
    NativeFloat(TensorData),
    NativeInt(TensorData),
}

impl RuntimeRecordValue {
    pub fn scalar<T: Serialize>(value: T) -> Self {
        Self::Scalar(bincode::serde::encode_to_vec(value, bincode::config::standard())
            .expect("Record scalar serialization failed"))
    }

    pub fn into_scalar<T: DeserializeOwned>(self) -> T {
        let Self::Scalar(bytes) = self else { panic!("Expected a scalar record field") };
        let (value, used) = bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
            .expect("Invalid record scalar");
        assert_eq!(used, bytes.len(), "Trailing bytes in record scalar");
        value
    }

    pub fn into_variant(self) -> (u32, Vec<Self>) {
        let Self::Fields(tag, values) = self else { panic!("Expected record fields") };
        (tag, values)
    }

    pub fn into_fields(self, tag: u32, count: usize) -> alloc::vec::IntoIter<Self> {
        let (actual, values) = self.into_variant();
        assert_eq!(tag, actual, "Record variant mismatch");
        assert_eq!(count, values.len(), "Record field count mismatch");
        values.into_iter()
    }

    pub fn convert_precision(self, precision: RuntimePrecision) -> Self {
        match self {
            Self::Float(data) | Self::NativeFloat(data) => {
                let data = precision.float(data);
                if precision.preserve_dtype { Self::NativeFloat(data) } else { Self::Float(data) }
            }
            Self::Int(data) | Self::NativeInt(data) => {
                let data = precision.int(data);
                if precision.preserve_dtype { Self::NativeInt(data) } else { Self::Int(data) }
            }
            Self::Fields(tag, values) => Self::Fields(tag, values.into_iter()
                .map(|value| value.convert_precision(precision)).collect()),
            Self::Adapter(name, value) => Self::Adapter(name, Box::new(value.convert_precision(precision))),
            value => value,
        }
    }

    pub(crate) fn encode(self) -> Vec<u8> {
        bincode::serde::encode_to_vec(self, bincode::config::standard())
            .expect("Reparameterization record serialization failed")
    }

    pub(crate) fn decode(bytes: &[u8]) -> Self {
        let (value, used) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())
            .expect("Invalid reparameterization record");
        assert_eq!(used, bytes.len(), "Trailing bytes in reparameterization record");
        value
    }
}

impl TransformRecord {
    pub(crate) fn from_runtime(name: String, value: RuntimeRecordValue, precision: Option<TypeId>) -> Self {
        Self { name, data: value.encode(), precision }
    }

    pub(crate) fn runtime_value(&self) -> RuntimeRecordValue { RuntimeRecordValue::decode(&self.data) }

    pub(crate) fn with_precision(self, precision: RuntimePrecision) -> Self {
        if self.precision == Some(precision.id) { return self; }
        let value = self.runtime_value().convert_precision(precision);
        Self::from_runtime(self.name, value, Some(precision.id))
    }
}

macro_rules! scalar_records {
    ($($ty:ty),* $(,)?) => {$(
        impl<B: Backend> RuntimeRecord<B> for $ty {
            fn into_runtime(self, _: RuntimePrecision) -> RuntimeRecordValue { RuntimeRecordValue::scalar(self) }
            fn from_runtime(value: RuntimeRecordValue, _: &B::Device) -> Self { value.into_scalar() }
        }
    )*};
}
scalar_records!((), String, bool, char, f64, f32, half::bf16, half::f16,
    usize, u128, u64, u32, u16, u8, isize, i128, i64, i32, i16, i8);

impl<B: Backend> RuntimeRecord<B> for EmptyRecord {
    fn into_runtime(self, _: RuntimePrecision) -> RuntimeRecordValue { RuntimeRecordValue::Empty }
    fn from_runtime(value: RuntimeRecordValue, _: &B::Device) -> Self {
        assert!(matches!(value, RuntimeRecordValue::Empty), "Expected an empty record");
        Self
    }
}

impl<B: Backend, const D: usize> RuntimeRecord<B> for Tensor<B, D> {
    fn into_runtime(self, precision: RuntimePrecision) -> RuntimeRecordValue {
        RuntimeRecordValue::Float(self.into_data()).convert_precision(precision)
    }
    fn from_runtime(value: RuntimeRecordValue, device: &B::Device) -> Self {
        let data = match value {
            RuntimeRecordValue::NativeFloat(data) => data,
            RuntimeRecordValue::Float(data) if matches!(data.dtype, DType::QFloat(_)) => data,
            RuntimeRecordValue::Float(data) => data.convert::<B::FloatElem>(),
            _ => panic!("Expected a floating tensor record"),
        };
        let dtype = data.dtype;
        Tensor::from_data(data, (device, dtype))
    }
}

impl<B: Backend, const D: usize> RuntimeRecord<B> for Tensor<B, D, Int> {
    fn into_runtime(self, precision: RuntimePrecision) -> RuntimeRecordValue {
        RuntimeRecordValue::Int(self.into_data()).convert_precision(precision)
    }
    fn from_runtime(value: RuntimeRecordValue, device: &B::Device) -> Self {
        let data = match value {
            RuntimeRecordValue::NativeInt(data) => data,
            RuntimeRecordValue::Int(data) => data.convert::<B::IntElem>(),
            _ => panic!("Expected an integer tensor record"),
        };
        let dtype = data.dtype;
        Tensor::from_data(data, (device, dtype))
    }
}

impl<B: Backend, const D: usize> RuntimeRecord<B> for Tensor<B, D, Bool> {
    fn into_runtime(self, _: RuntimePrecision) -> RuntimeRecordValue { RuntimeRecordValue::Bool(self.into_data()) }
    fn from_runtime(value: RuntimeRecordValue, device: &B::Device) -> Self {
        let RuntimeRecordValue::Bool(data) = value else { panic!("Expected a boolean tensor record") };
        Tensor::from_data(data, device)
    }
}

impl<B: Backend, const D: usize> Param<Tensor<B, D>> {
    pub(crate) fn encode_transform(&self, precision: RuntimePrecision) -> Option<TransformRecord> {
        match &self.transform {
            Some(transform) => Some(transform.record(precision)),
            None => self.transform_record.clone().map(|record| record.with_precision(precision)),
        }
    }
}

impl<B: Backend, const D: usize> RuntimeRecord<B> for Param<Tensor<B, D>> {
    fn into_runtime(self, precision: RuntimePrecision) -> RuntimeRecordValue {
        let transform = self.encode_transform(precision).map(|record|
            RuntimeRecordValue::Adapter(record.name.clone(), Box::new(record.runtime_value())))
            .unwrap_or(RuntimeRecordValue::Empty);
        let (id, tensor, mapper) = self.consume();
        RuntimeRecordValue::Fields(0, vec![RuntimeRecordValue::scalar(id.serialize()),
            mapper.on_save(tensor).into_runtime(precision), transform])
    }
    fn from_runtime(value: RuntimeRecordValue, device: &B::Device) -> Self {
        B::memory_persistent_allocations(device, value, |value| {
            let mut fields = value.into_fields(0, 3);
            let id: String = fields.next().unwrap().into_scalar();
            let tensor = Tensor::<B, D>::from_runtime(fields.next().unwrap(), device);
            let mut param = Param::initialized(ParamId::deserialize(&id), Parameter::set_require_grad(tensor, true));
            param.require_grad = true;
            param.transform_record = match fields.next().unwrap() {
                RuntimeRecordValue::Empty => None,
                RuntimeRecordValue::Adapter(name, value) => Some(TransformRecord::from_runtime(name, *value, None)),
                _ => panic!("Invalid parameter reparameterization record"),
            };
            param
        })
    }
}

macro_rules! parameter_record {
    ($kind:ty) => {
        impl<B: Backend, const D: usize> RuntimeRecord<B> for Param<Tensor<B, D, $kind>> {
            fn into_runtime(self, precision: RuntimePrecision) -> RuntimeRecordValue {
                let (id, tensor, mapper) = self.consume();
                RuntimeRecordValue::Fields(0, vec![RuntimeRecordValue::scalar(id.serialize()),
                    mapper.on_save(tensor).into_runtime(precision)])
            }
            fn from_runtime(value: RuntimeRecordValue, device: &B::Device) -> Self {
                B::memory_persistent_allocations(device, value, |value| {
                    let mut fields = value.into_fields(0, 2);
                    let id: String = fields.next().unwrap().into_scalar();
                    Param::initialized(ParamId::deserialize(&id), Tensor::from_runtime(fields.next().unwrap(), device))
                })
            }
        }
    };
}
parameter_record!(Int);
parameter_record!(Bool);

impl<B: ruda_tensor::ops::SparseOps> RuntimeRecord<B> for super::CsrTensorRecord<B>
where B::CsrData: Serialize + DeserializeOwned {
    fn into_runtime(self, _: RuntimePrecision) -> RuntimeRecordValue { RuntimeRecordValue::scalar(self.into_data()) }
    fn from_runtime(value: RuntimeRecordValue, _: &B::Device) -> Self { Self::from_data(value.into_scalar()) }
}
