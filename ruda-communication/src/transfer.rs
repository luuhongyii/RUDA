use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorTransferId(u64);

impl From<u64> for TensorTransferId {
    fn from(value: u64) -> Self { Self(value) }
}
impl TensorTransferId {
    pub fn next(&mut self) { self.0 += 1; }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TransferCapability([u8; 32]);
impl TransferCapability {
    pub fn random() -> Self { Self(ruda_core::rand::gen_random()) }
}
impl core::fmt::Debug for TransferCapability {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { f.write_str("TransferCapability([redacted])") }
}
