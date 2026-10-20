use alloc::string::{String, ToString};

/// Where a [`Store`](super::Store)'s entries live inside an environment: a
/// `/`-separated location such as `autotune/0.11.0/cuda-0/matmul`.
///
/// The middle segment is this build's version, and the constructors inject it
/// unconditionally: it is what makes entries written by one ruda invisible
/// to another, so a bundle built elsewhere can't be read as if it matched.
/// The [`From`] impls are the escape hatch that skips versioning, for callers
/// addressing an explicit [`Storage`](super::Storage) directly.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Namespace {
    full: String,
}

/// The name used when none is given.
const DEFAULT_NAME: &str = "ruda";

impl Namespace {
    /// The namespace for `path` under the default name:
    /// `ruda/<version>/<path>`.
    pub fn new<P: AsRef<str>>(path: P) -> Self {
        Self::scoped(DEFAULT_NAME, path)
    }

    /// The namespace for `path` under `name`: `<name>/<version>/<path>`.
    pub fn scoped<N: AsRef<str>, P: AsRef<str>>(name: N, path: P) -> Self {
        let version = env!("CARGO_PKG_VERSION");
        let name = name.as_ref();
        let path = path.as_ref().trim_matches('/');

        Self {
            full: alloc::format!("{name}/{version}/{path}"),
        }
    }

    /// The full `/`-separated namespace.
    pub fn as_str(&self) -> &str {
        &self.full
    }
}

/// Verbatim, with no version segment injected.
impl From<String> for Namespace {
    fn from(full: String) -> Self {
        Self { full }
    }
}

/// Verbatim, with no version segment injected.
impl From<&str> for Namespace {
    fn from(full: &str) -> Self {
        full.to_string().into()
    }
}

impl core::fmt::Display for Namespace {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.full)
    }
}
