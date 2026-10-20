use alloc::string::String;
use core::{fmt, marker::PhantomData};
use serde::{Serialize, Deserialize, Serializer, Deserializer,
    ser::SerializeStruct, de::{Error, MapAccess, SeqAccess, Visitor}};
use crate::module::TransformRecord;

const EXTENDED_ID: &str = "ruda-param-v1:";

/// A parameter record. Adapter-free values retain the original two-field encoding.
#[derive(Debug, Clone)]
pub struct ParamSerde<T> {
    pub(super) id: String,
    pub(super) param: T,
    pub(super) reparameterization: Option<TransformRecord>,
}

impl<T> ParamSerde<T> {
    /// Create an adapter-free parameter record.
    pub fn new(id: String, param: T) -> Self { Self { id, param, reparameterization: None } }
    pub(super) fn with_transform(mut self, transform: Option<TransformRecord>) -> Self {
        self.reparameterization = transform;
        self
    }
}

impl<T: Serialize> Serialize for ParamSerde<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut record = serializer.serialize_struct("ParamSerde", if self.reparameterization.is_some() { 3 } else { 2 })?;
        match &self.reparameterization {
            Some(transform) => {
                record.serialize_field("id", &alloc::format!("{EXTENDED_ID}{}", self.id))?;
                record.serialize_field("param", &self.param)?;
                record.serialize_field("reparameterization", transform)?;
            }
            None => {
                record.serialize_field("id", &self.id)?;
                record.serialize_field("param", &self.param)?;
            }
        }
        record.end()
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for ParamSerde<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ParamVisitor<T>(PhantomData<T>);
        impl<'de, T: Deserialize<'de>> Visitor<'de> for ParamVisitor<T> {
            type Value = ParamSerde<T>;
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result { formatter.write_str("a parameter record") }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let id: String = seq.next_element()?.ok_or_else(|| A::Error::missing_field("id"))?;
                let param = seq.next_element()?.ok_or_else(|| A::Error::missing_field("param"))?;
                if let Some(id) = id.strip_prefix(EXTENDED_ID) {
                    let reparameterization = seq.next_element()?.ok_or_else(|| A::Error::missing_field("reparameterization"))?;
                    Ok(ParamSerde { id: String::from(id), param, reparameterization: Some(reparameterization) })
                } else { Ok(ParamSerde::new(id, param)) }
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let (mut id, mut param, mut reparameterization) = (None, None, None);
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "id" => {
                            if id.is_some() { return Err(A::Error::duplicate_field("id")); }
                            id = Some(map.next_value::<String>()?);
                        }
                        "param" => {
                            if param.is_some() { return Err(A::Error::duplicate_field("param")); }
                            param = Some(map.next_value()?);
                        }
                        "reparameterization" => {
                            if reparameterization.is_some() { return Err(A::Error::duplicate_field("reparameterization")); }
                            reparameterization = Some(map.next_value()?);
                        }
                        _ => { let _ = map.next_value::<serde::de::IgnoredAny>()?; }
                    }
                }
                let id = id.ok_or_else(|| A::Error::missing_field("id"))?;
                let id = match id.strip_prefix(EXTENDED_ID) {
                    Some(original) => {
                        if reparameterization.is_none() { return Err(A::Error::missing_field("reparameterization")); }
                        String::from(original)
                    }
                    None => id,
                };
                Ok(ParamSerde { id, param: param.ok_or_else(|| A::Error::missing_field("param"))?, reparameterization })
            }
        }
        deserializer.deserialize_struct("ParamSerde", &["id", "param", "reparameterization"], ParamVisitor(PhantomData))
    }
}
