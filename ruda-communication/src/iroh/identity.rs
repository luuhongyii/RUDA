#[cfg(not(target_family = "wasm"))]
use std::{fs, io::{self, ErrorKind, Write}, path::Path};

#[derive(Clone)]
pub struct IrohIdentity(::iroh::SecretKey);

impl IrohIdentity {
    pub fn random() -> Self { Self(::iroh::SecretKey::generate()) }
    pub fn from_bytes(bytes: [u8; 32]) -> Self { Self(::iroh::SecretKey::from_bytes(&bytes)) }
    pub fn to_bytes(&self) -> [u8; 32] { self.0.to_bytes() }
    pub fn id(&self) -> ::iroh::EndpointId { self.0.public() }
    pub(crate) fn secret_key(&self) -> ::iroh::SecretKey { self.0.clone() }

    #[cfg(not(target_family = "wasm"))]
    pub fn load_or_create(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        match Self::load(path) {
            Err(error) if error.kind() == ErrorKind::NotFound => {
                let dir = path.parent().filter(|dir| !dir.as_os_str().is_empty()).unwrap_or(Path::new("."));
                let identity = Self::random();
                let mut staged = tempfile::NamedTempFile::new_in(dir)?;
                staged.write_all(&identity.to_bytes())?;
                staged.as_file().sync_all()?;
                match staged.persist_noclobber(path) {
                    Ok(_) => Ok(identity),
                    Err(error) if error.error.kind() == ErrorKind::AlreadyExists => Self::load(path),
                    Err(error) => Err(error.error),
                }
            }
            result => result,
        }
    }

    #[cfg(not(target_family = "wasm"))]
    fn load(path: &Path) -> io::Result<Self> {
        fs::read(path)?.try_into().map(Self::from_bytes)
            .map_err(|_| io::Error::new(ErrorKind::InvalidData, "An Iroh identity must contain exactly 32 bytes"))
    }
}

impl core::fmt::Debug for IrohIdentity {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("IrohIdentity").field("id", &self.id()).finish_non_exhaustive()
    }
}
