use alloc::string::String;
use core::{fmt, str::FromStr};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct DeviceIdentity {
    pub name: String,
    pub fingerprint: String,
    pub physical: Option<PhysicalDevice>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub struct PhysicalDevice {
    pub pci_address: Option<PciAddress>,
    pub luid: Option<AdapterLuid>,
    pub registry_entry_id: Option<RegistryEntryId>,
    pub vendor: Option<PciVendor>,
}

impl PhysicalDevice {
    pub fn is_same_card(&self, other: &Self) -> bool {
        if let (Some(a), Some(b)) = (self.pci_address, other.pci_address) {
            return a == b;
        }
        if let (Some(a), Some(b)) = (self.luid, other.luid) {
            return a == b;
        }
        matches!((self.registry_entry_id, other.registry_entry_id), (Some(a), Some(b)) if a == b)
    }

    pub fn fill_from(&mut self, other: &Self) {
        self.pci_address = self.pci_address.or(other.pci_address);
        self.luid = self.luid.or(other.luid);
        self.registry_entry_id = self.registry_entry_id.or(other.registry_entry_id);
        self.vendor = self.vendor.or(other.vendor);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegistryEntryId(u64);

impl RegistryEntryId {
    pub fn new(id: u64) -> Self { Self(id) }
    pub fn get(self) -> u64 { self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AdapterLuid([u8; 8]);

impl AdapterLuid {
    pub fn new(bytes: [u8; 8]) -> Self { Self(bytes) }
    pub fn from_parts(low_part: u32, high_part: i32) -> Self {
        let mut bytes = [0; 8];
        bytes[..4].copy_from_slice(&low_part.to_le_bytes());
        bytes[4..].copy_from_slice(&high_part.to_le_bytes());
        Self(bytes)
    }
    pub fn bytes(self) -> [u8; 8] { self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PciVendor { Nvidia, Amd, Intel, Apple, Arm, Qualcomm, Other(u32) }

impl PciVendor {
    pub fn id(self) -> u32 {
        match self {
            Self::Nvidia => 0x10de,
            Self::Amd => 0x1002,
            Self::Intel => 0x8086,
            Self::Apple => 0x106b,
            Self::Arm => 0x13b5,
            Self::Qualcomm => 0x5143,
            Self::Other(id) => id,
        }
    }
}

impl From<u32> for PciVendor {
    fn from(id: u32) -> Self {
        match id {
            0x10de => Self::Nvidia,
            0x1002 => Self::Amd,
            0x8086 => Self::Intel,
            0x106b => Self::Apple,
            0x13b5 => Self::Arm,
            0x5143 => Self::Qualcomm,
            id => Self::Other(id),
        }
    }
}

impl fmt::Display for PciVendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nvidia => f.write_str("NVIDIA"),
            Self::Amd => f.write_str("AMD"),
            Self::Intel => f.write_str("Intel"),
            Self::Apple => f.write_str("Apple"),
            Self::Arm => f.write_str("Arm"),
            Self::Qualcomm => f.write_str("Qualcomm"),
            Self::Other(id) => write!(f, "{id:#06x}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PciAddress {
    pub domain: u32,
    pub bus: u8,
    pub device: u8,
    pub function: u8,
}

impl fmt::Display for PciAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04x}:{:02x}:{:02x}.{:x}", self.domain, self.bus, self.device, self.function)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PciAddressError(pub String);

impl fmt::Display for PciAddressError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "not a PCI address: {}", self.0) }
}

impl core::error::Error for PciAddressError {}

impl FromStr for PciAddress {
    type Err = PciAddressError;
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let err = || PciAddressError(String::from(text));
        let (rest, function) = text.rsplit_once('.').ok_or_else(err)?;
        let mut parts = rest.rsplitn(3, ':');
        let device = parts.next().ok_or_else(err)?;
        let bus = parts.next().ok_or_else(err)?;
        let domain = parts.next().unwrap_or("0");
        Ok(Self {
            domain: u32::from_str_radix(domain, 16).map_err(|_| err())?,
            bus: u8::from_str_radix(bus, 16).map_err(|_| err())?,
            device: u8::from_str_radix(device, 16).map_err(|_| err())?,
            function: u8::from_str_radix(function, 16).map_err(|_| err())?,
        })
    }
}
