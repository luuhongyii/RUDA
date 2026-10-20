use std::{fmt, sync::Arc};

/// Opaque application credential. Debug formatting never includes its bytes.
#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub struct Credential(Arc<[u8]>);

impl Credential {
    pub fn as_bytes(&self) -> &[u8] { &self.0 }
    pub fn is_empty(&self) -> bool { self.0.is_empty() }
}
impl fmt::Debug for Credential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str("Credential(..)") }
}
impl From<&[u8]> for Credential { fn from(value: &[u8]) -> Self { Self(value.into()) } }
impl<const N: usize> From<&[u8; N]> for Credential { fn from(value: &[u8; N]) -> Self { Self(value.as_slice().into()) } }
impl From<Vec<u8>> for Credential { fn from(value: Vec<u8>) -> Self { Self(value.into()) } }
impl From<&str> for Credential { fn from(value: &str) -> Self { Self(value.as_bytes().into()) } }
impl From<String> for Credential { fn from(value: String) -> Self { Self(value.into_bytes().into()) } }
impl From<&String> for Credential { fn from(value: &String) -> Self { value.as_str().into() } }
