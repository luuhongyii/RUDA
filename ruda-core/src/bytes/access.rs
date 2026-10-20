use alloc::string::String;

/// Whether an access may materialize lazy storage or copy shared storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccessPolicy {
    allow_copy: bool,
}

impl Default for AccessPolicy {
    fn default() -> Self {
        Self::allow_copy()
    }
}

impl AccessPolicy {
    /// Allow materialization and copy-on-write.
    pub fn allow_copy() -> Self {
        Self { allow_copy: true }
    }

    /// Require already-resident, appropriately owned storage.
    pub fn zero_copy() -> Self {
        Self { allow_copy: false }
    }

    /// Whether materialization or copying is permitted.
    pub fn copy_allowed(&self) -> bool {
        self.allow_copy
    }
}

/// An error accessing a byte allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessError {
    /// The allocation cannot satisfy the requested access without copying.
    WouldCopy,
    /// Materializing the allocation failed.
    Read(String),
}

impl core::fmt::Display for AccessError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::WouldCopy => f.write_str("access would require a copy, which the policy forbids"),
            Self::Read(message) => write!(f, "failed to materialize bytes: {message}"),
        }
    }
}

impl core::error::Error for AccessError {}

/// Configures immutable byte access.
#[derive(Clone, Copy, Debug, Default)]
pub struct Reader {
    pub(crate) policy: AccessPolicy,
}

impl Reader {
    /// Allow materialization on read.
    pub fn new() -> Self {
        Self::default()
    }

    /// Require resident storage without materialization.
    pub fn no_copy(mut self) -> Self {
        self.policy = AccessPolicy::zero_copy();
        self
    }
}

/// Configures mutable byte access.
#[derive(Clone, Copy, Debug, Default)]
pub struct Writer {
    pub(crate) policy: AccessPolicy,
}

impl Writer {
    /// Allow materialization and copy-on-write.
    pub fn new() -> Self {
        Self::default()
    }

    /// Require already-resident private storage.
    pub fn no_copy(mut self) -> Self {
        self.policy = AccessPolicy::zero_copy();
        self
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::bytes::Bytes;

    #[test]
    fn resident_storage_supports_no_copy_read_and_write() {
        let mut bytes = Bytes::from_bytes_vec(alloc::vec![1, 2, 3]);
        assert_eq!(bytes.read(Reader::new().no_copy()).unwrap(), [1, 2, 3]);
        bytes.write(Writer::new().no_copy()).unwrap()[0] = 9;
        assert_eq!(bytes.read(Reader::new()).unwrap(), [9, 2, 3]);
    }

    #[cfg(feature = "shared-bytes")]
    #[test]
    fn shared_storage_honors_no_copy_before_and_after_detachment() {
        let source = bytes::Bytes::from_static(&[1, 2, 3]);
        let mut bytes = Bytes::from_shared(source.clone(), crate::bytes::AllocationProperty::Other);
        assert_eq!(bytes.read(Reader::new().no_copy()).unwrap(), [1, 2, 3]);
        assert!(matches!(
            bytes.write(Writer::new().no_copy()),
            Err(AccessError::WouldCopy)
        ));
        bytes.write(Writer::new()).unwrap()[0] = 9;
        bytes.write(Writer::new().no_copy()).unwrap()[1] = 8;
        assert_eq!(bytes.read(Reader::new().no_copy()).unwrap(), [9, 8, 3]);
        assert_eq!(&source[..], [1, 2, 3]);
    }

    #[test]
    fn file_no_copy_capacity_and_failed_read_retry() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("missing");
        let mut bytes = Bytes::from_file(&path, 3, 0);
        assert_eq!(bytes.capacity(), 3);
        assert_eq!(bytes.len(), 3);
        assert!(matches!(
            bytes.read(Reader::new().no_copy()),
            Err(AccessError::WouldCopy)
        ));
        assert!(matches!(
            bytes.write(Writer::new().no_copy()),
            Err(AccessError::WouldCopy)
        ));
        assert!(matches!(
            bytes.read(Reader::new()),
            Err(AccessError::Read(_))
        ));
        assert!(matches!(
            bytes.write(Writer::new()),
            Err(AccessError::Read(_))
        ));
        std::fs::write(&path, [1, 2, 3]).unwrap();
        assert_eq!(bytes.read(Reader::new()).unwrap(), [1, 2, 3]);
        bytes.write(Writer::new().no_copy()).unwrap()[0] = 9;
        assert_eq!(bytes.read(Reader::new().no_copy()).unwrap(), [9, 2, 3]);
        assert_eq!(std::fs::read(&path).unwrap(), [1, 2, 3]);
    }

    #[test]
    fn truncated_file_materialization_can_retry() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("short");
        std::fs::write(&path, [1, 2]).unwrap();
        let bytes = Bytes::from_file(&path, 3, 0);
        assert!(matches!(
            bytes.read(Reader::new()),
            Err(AccessError::Read(_))
        ));
        std::fs::write(&path, [1, 2, 3]).unwrap();
        assert_eq!(bytes.read(Reader::new()).unwrap(), [1, 2, 3]);
    }

    #[test]
    fn concurrent_file_reads_share_one_materialized_buffer() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("shared");
        std::fs::write(&path, [1, 2, 3]).unwrap();
        let bytes = Bytes::from_file(&path, 3, 0);
        let barrier = std::sync::Barrier::new(8);
        let pointers = std::thread::scope(|scope| {
            let readers: alloc::vec::Vec<_> = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        let data = bytes.read(Reader::new()).unwrap();
                        assert_eq!(data, [1, 2, 3]);
                        data.as_ptr() as usize
                    })
                })
                .collect();
            readers
                .into_iter()
                .map(|reader| reader.join().unwrap())
                .collect::<alloc::vec::Vec<_>>()
        });
        assert!(pointers.iter().all(|pointer| *pointer == pointers[0]));
    }
}
