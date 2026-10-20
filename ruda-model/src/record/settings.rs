use ruda_tensor::api::Element;
use serde::{Serialize, de::DeserializeOwned};

/// Settings allowing to control the precision when (de)serializing items.
pub trait PrecisionSettings:
    Send + Sync + core::fmt::Debug + core::default::Default + Clone
{
    /// Float element type.
    type FloatElem: Element + Serialize + DeserializeOwned;

    /// Integer element type.
    type IntElem: Element + Serialize + DeserializeOwned;

    /// Retain each tensor's original storage dtype instead of converting it.
    const PRESERVE_DTYPE: bool = false;
}

/// Default precision settings.
#[derive(Debug, Default, Clone)]
pub struct FullPrecisionSettings;

/// Precision settings optimized for compactness.
#[derive(Debug, Default, Clone)]
pub struct HalfPrecisionSettings;

/// Precision settings optimized for precision.
#[derive(Debug, Default, Clone)]
pub struct DoublePrecisionSettings;

/// Lossless tensor recording, including mixed storage dtypes.
#[derive(Debug, Default, Clone)]
pub struct NativePrecisionSettings;

impl PrecisionSettings for NativePrecisionSettings {
    type FloatElem = f64;
    type IntElem = i64;
    const PRESERVE_DTYPE: bool = true;
}

impl PrecisionSettings for FullPrecisionSettings {
    type FloatElem = f32;
    type IntElem = i32;
}

impl PrecisionSettings for DoublePrecisionSettings {
    type FloatElem = f64;
    type IntElem = i64;
}

impl PrecisionSettings for HalfPrecisionSettings {
    type FloatElem = half::f16;
    type IntElem = i16;
}
