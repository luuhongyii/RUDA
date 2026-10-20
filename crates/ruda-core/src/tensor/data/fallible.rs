use super::{DataError, TensorData};
use crate::bytes::{Reader, Writer};
use crate::tensor::{
    BoolStore, DType,
    element::{Element, ElementConversion},
};
use alloc::{vec, vec::Vec};
use bytemuck::{AnyBitPattern, CheckedBitPattern, Zeroable, cast_mut};
use half::{bf16, f16};

impl TensorData {
    /// Copy stored elements without dtype conversion, propagating storage errors.
    pub fn try_to_vec<E: Element>(&self) -> Result<Vec<E>, DataError> {
        if !self.matches_target_dtype::<E>() {
            return Err(DataError::DTypeMismatch {
                expected: E::dtype(),
                actual: self.dtype,
            });
        }
        Ok(
            bytemuck::checked::try_cast_slice(self.bytes.read(Reader::new())?)
                .map_err(DataError::InvalidRepresentation)?
                .to_vec(),
        )
    }

    /// Read stored elements without dtype conversion, reusing ownership when possible.
    pub fn try_into_vec<E: Element>(self) -> Result<Vec<E>, DataError> {
        if !self.matches_target_dtype::<E>() {
            return Err(DataError::DTypeMismatch {
                expected: E::dtype(),
                actual: self.dtype,
            });
        }
        self.try_read_vec_unchecked()
    }

    fn try_read_vec_unchecked<E: Element>(mut self) -> Result<Vec<E>, DataError> {
        self.bytes.read(Reader::new())?;
        self.bytes = match self.bytes.try_into_vec::<E>() {
            Ok(values) => return Ok(values),
            Err(bytes) => bytes,
        };
        Ok(
            bytemuck::checked::try_cast_slice(self.bytes.read(Reader::new())?)
                .map_err(DataError::InvalidRepresentation)?
                .to_vec(),
        )
    }

    /// Copy and convert elements to the requested host element type.
    pub fn try_to_vec_as<E: Element>(&self) -> Result<Vec<E>, DataError> {
        self.clone().try_into_vec_as::<E>()
    }

    /// Read and convert elements to the requested host element type.
    pub fn try_into_vec_as<E: Element>(self) -> Result<Vec<E>, DataError> {
        self.try_cast_as::<E>()?.try_read_vec_unchecked()
    }

    /// Convert to the dtype represented by `E`, returning storage and conversion errors.
    pub fn try_cast_as<E: Element>(self) -> Result<Self, DataError> {
        self.try_cast(E::dtype())
    }

    /// Convert dtype; identical dtypes are returned unchanged without reading storage.
    pub fn try_cast(self, dtype: DType) -> Result<Self, DataError> {
        if dtype == self.dtype {
            return Ok(self);
        }
        if dtype.size() == self.dtype.size()
            && !matches!(
                self.dtype,
                DType::Bool(BoolStore::Native) | DType::QFloat(_)
            )
            && !matches!(dtype, DType::Bool(BoolStore::Native) | DType::QFloat(_))
        {
            match self.dtype {
                DType::F64 => self.try_cast_inplace_from::<f64>(dtype),
                DType::F32 | DType::Flex32 => self.try_cast_inplace_from::<f32>(dtype),
                DType::F16 => self.try_cast_inplace_from::<f16>(dtype),
                DType::BF16 => self.try_cast_inplace_from::<bf16>(dtype),
                DType::I64 => self.try_cast_inplace_from::<i64>(dtype),
                DType::I32 => self.try_cast_inplace_from::<i32>(dtype),
                DType::I16 => self.try_cast_inplace_from::<i16>(dtype),
                DType::I8 => self.try_cast_inplace_from::<i8>(dtype),
                DType::U64 => self.try_cast_inplace_from::<u64>(dtype),
                DType::U32 | DType::Bool(BoolStore::U32) => {
                    self.try_cast_inplace_from::<u32>(dtype)
                }
                DType::U16 => self.try_cast_inplace_from::<u16>(dtype),
                DType::U8 | DType::Bool(BoolStore::U8) => self.try_cast_inplace_from::<u8>(dtype),
                DType::Bool(BoolStore::Native) | DType::QFloat(_) => {
                    Err(DataError::DTypeMismatch {
                        expected: dtype,
                        actual: self.dtype,
                    })
                }
            }
        } else {
            match self.dtype {
                DType::F64 => self.try_cast_clone_from::<f64>(dtype),
                DType::F32 | DType::Flex32 => self.try_cast_clone_from::<f32>(dtype),
                DType::F16 => self.try_cast_clone_from::<f16>(dtype),
                DType::BF16 => self.try_cast_clone_from::<bf16>(dtype),
                DType::I64 => self.try_cast_clone_from::<i64>(dtype),
                DType::I32 => self.try_cast_clone_from::<i32>(dtype),
                DType::I16 => self.try_cast_clone_from::<i16>(dtype),
                DType::I8 => self.try_cast_clone_from::<i8>(dtype),
                DType::U64 => self.try_cast_clone_from::<u64>(dtype),
                DType::U32 | DType::Bool(BoolStore::U32) => self.try_cast_clone_from::<u32>(dtype),
                DType::U16 => self.try_cast_clone_from::<u16>(dtype),
                DType::U8 | DType::Bool(BoolStore::U8) => self.try_cast_clone_from::<u8>(dtype),
                DType::Bool(BoolStore::Native) => self.try_cast_clone_from::<bool>(dtype),
                DType::QFloat(_) => Err(DataError::UnsupportedConversion {
                    from: self.dtype,
                    to: dtype,
                }),
            }
        }
    }

    fn try_cast_inplace_from<Current: Element + AnyBitPattern>(
        self,
        dtype: DType,
    ) -> Result<Self, DataError> {
        match dtype {
            DType::F64 => self.try_convert_inplace::<Current, f64>(|x| x.elem()),
            DType::F32 | DType::Flex32 => self.try_convert_inplace::<Current, f32>(|x| x.elem()),
            DType::F16 => self.try_convert_inplace::<Current, f16>(|x| x.elem()),
            DType::BF16 => self.try_convert_inplace::<Current, bf16>(|x| x.elem()),
            DType::I64 => self.try_convert_inplace::<Current, i64>(|x| x.elem()),
            DType::I32 => self.try_convert_inplace::<Current, i32>(|x| x.elem()),
            DType::I16 => self.try_convert_inplace::<Current, i16>(|x| x.elem()),
            DType::I8 => self.try_convert_inplace::<Current, i8>(|x| x.elem()),
            DType::U64 => self.try_convert_inplace::<Current, u64>(|x| x.elem()),
            DType::U32 => self.try_convert_inplace::<Current, u32>(|x| x.elem()),
            DType::U16 => self.try_convert_inplace::<Current, u16>(|x| x.elem()),
            DType::U8 => self.try_convert_inplace::<Current, u8>(|x| x.elem()),
            DType::Bool(BoolStore::U8) => self
                .try_convert_inplace::<Current, u8>(|x| x.to_bool().elem())
                .map(TensorData::into_bool_u8),
            DType::Bool(BoolStore::U32) => self
                .try_convert_inplace::<Current, u32>(|x| x.to_bool().elem())
                .map(TensorData::into_bool_u32),
            DType::Bool(BoolStore::Native) | DType::QFloat(_) => Err(DataError::DTypeMismatch {
                expected: dtype,
                actual: Current::dtype(),
            }),
        }
    }

    fn try_convert_inplace<Current: Element + AnyBitPattern, Target: Element + AnyBitPattern>(
        mut self,
        transform: impl Fn(&Current) -> Target,
    ) -> Result<Self, DataError> {
        let expected = self.num_elements();
        let values =
            bytemuck::checked::try_cast_slice_mut::<_, Current>(self.bytes.write(Writer::new())?)
                .map_err(DataError::InvalidRepresentation)?;
        let actual = values.len();
        if actual != expected {
            return Err(DataError::ElementCountMismatch { expected, actual });
        }
        for value in values {
            let target = transform(value);
            *cast_mut::<_, Target>(value) = target;
        }
        self.dtype = Target::dtype();
        Ok(self)
    }

    fn try_cast_clone_from<Current: Element + CheckedBitPattern>(
        self,
        dtype: DType,
    ) -> Result<Self, DataError> {
        match dtype {
            DType::F64 => self.try_convert_clone::<Current, f64>(|x| x.elem()),
            DType::F32 | DType::Flex32 => self.try_convert_clone::<Current, f32>(|x| x.elem()),
            DType::F16 => self.try_convert_clone::<Current, f16>(|x| x.elem()),
            DType::BF16 => self.try_convert_clone::<Current, bf16>(|x| x.elem()),
            DType::I64 => self.try_convert_clone::<Current, i64>(|x| x.elem()),
            DType::I32 => self.try_convert_clone::<Current, i32>(|x| x.elem()),
            DType::I16 => self.try_convert_clone::<Current, i16>(|x| x.elem()),
            DType::I8 => self.try_convert_clone::<Current, i8>(|x| x.elem()),
            DType::U64 => self.try_convert_clone::<Current, u64>(|x| x.elem()),
            DType::U32 => self.try_convert_clone::<Current, u32>(|x| x.elem()),
            DType::U16 => self.try_convert_clone::<Current, u16>(|x| x.elem()),
            DType::U8 => self.try_convert_clone::<Current, u8>(|x| x.elem()),
            DType::Bool(BoolStore::Native) => self.try_convert_clone::<Current, bool>(|x| x.elem()),
            DType::Bool(BoolStore::U8) => self
                .try_convert_clone::<Current, u8>(|x| x.to_bool().elem())
                .map(TensorData::into_bool_u8),
            DType::Bool(BoolStore::U32) => self
                .try_convert_clone::<Current, u32>(|x| x.to_bool().elem())
                .map(TensorData::into_bool_u32),
            DType::QFloat(_) => Err(DataError::UnsupportedConversion {
                from: self.dtype,
                to: dtype,
            }),
        }
    }

    fn try_convert_clone<Current: Element + CheckedBitPattern, Target: Element + Zeroable>(
        self,
        transform: impl Fn(&Current) -> Target,
    ) -> Result<Self, DataError> {
        let expected = self.num_elements();
        let values =
            bytemuck::checked::try_cast_slice::<_, Current>(self.bytes.read(Reader::new())?)
                .map_err(DataError::InvalidRepresentation)?;
        let actual = values.len();
        if actual != expected {
            return Err(DataError::ElementCountMismatch { expected, actual });
        }
        let mut output: Vec<Target> = vec![Zeroable::zeroed(); expected];
        for (value, output) in values.iter().zip(&mut output) {
            *output = transform(value);
        }
        Ok(Self::new(output, self.shape))
    }
}
