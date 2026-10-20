use alloc::{format, string::{String, ToString}, vec::Vec};
use hashbrown::HashMap;
use ruda_model::module::{Module, ModuleMapper, ModuleVisitor, Param, ParamGroup, ParamId};
use ruda_tensor::api::{backend::Backend, BasicOps, Bool, DType, Float, Int, Shape, Tensor, TensorData};

/// Dtype selection when applying a path-addressed module record.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DTypePolicy {
    /// Load the recorded storage dtype without conversion.
    #[default]
    FromRecord,
    /// Cast each matching tensor to the destination parameter's current dtype.
    CastToModule,
}

/// A record could not be read, written, or applied to the destination module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModuleRecordError {
    /// Storage or tensor-data read failure.
    Io(String),
    /// Incompatible shapes, missing parameters, or unused record entries.
    Validation(String),
}

impl core::fmt::Display for ModuleRecordError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Io(message) => write!(f, "Record I/O error: {message}"),
            Self::Validation(message) => write!(f, "Record validation error: {message}"),
        }
    }
}
impl core::error::Error for ModuleRecordError {}

#[cfg(feature = "rudapack")]
impl From<crate::RudapackError> for ModuleRecordError {
    fn from(error: crate::RudapackError) -> Self { Self::Io(error.to_string()) }
}

#[derive(Clone)]
struct RecordTensor {
    path: String,
    id: ParamId,
    data: TensorData,
}

/// Backend-independent parameter data, addressed by full module paths.
/// Runtime flags and model configuration remain owned by the destination module.
#[derive(Clone)]
pub struct ModuleRecord {
    tensors: Vec<RecordTensor>,
    dtype_policy: DTypePolicy,
    allow_partial: bool,
    allow_unused: bool,
    validate: bool,
}

impl core::fmt::Debug for ModuleRecord {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ModuleRecord")
            .field("num_tensors", &self.tensors.len())
            .field("dtype_policy", &self.dtype_policy)
            .field("allow_partial", &self.allow_partial)
            .field("allow_unused", &self.allow_unused)
            .field("validate", &self.validate).finish()
    }
}

impl ModuleRecord {
    fn from_tensors(tensors: Vec<RecordTensor>) -> Self {
        Self { tensors, dtype_policy: DTypePolicy::default(), allow_partial: false,
            allow_unused: false, validate: true }
    }

    /// Collect saved parameter values, excluding unselected values before initialization/readback.
    pub fn from_module<B: Backend, M: Module<B>>(module: &M, group: Option<ParamGroup>) -> Self {
        let mut collector = RecordCollector { path: Vec::new(), group, tensors: Vec::new() };
        module.visit(&mut collector);
        Self::from_tensors(collector.tensors)
    }

    /// Number of recorded parameter paths.
    pub fn len(&self) -> usize { self.tensors.len() }

    /// Whether the record contains no tensors.
    pub fn is_empty(&self) -> bool { self.tensors.is_empty() }

    /// Select dtype behavior for loading; saving retains each recorded dtype.
    pub fn with_dtype_policy(mut self, policy: DTypePolicy) -> Self { self.dtype_policy = policy; self }

    /// Cast matching data to destination parameter dtypes when loading.
    pub fn cast_to_module_dtype(self) -> Self { self.with_dtype_policy(DTypePolicy::CastToModule) }

    /// Permit destination parameter paths absent from the record.
    pub fn allow_partial(mut self, allow: bool) -> Self { self.allow_partial = allow; self }

    /// Permit record paths absent from the destination module.
    pub fn allow_unused(mut self, allow: bool) -> Self { self.allow_unused = allow; self }

    /// Report shape failures, or leave mismatched destination parameters unchanged.
    pub fn validate(mut self, validate: bool) -> Self { self.validate = validate; self }

    /// Apply data and persisted parameter IDs, retaining destination configuration and flags.
    pub fn apply<B: Backend, M: Module<B>>(self, module: M) -> Result<M, ModuleRecordError> {
        let (validate, allow_partial, allow_unused) = (self.validate, self.allow_partial, self.allow_unused);
        let mut mapper = RecordMapper {
            path: Vec::new(),
            tensors: self.tensors.into_iter().map(|tensor| (tensor.path, (tensor.id, tensor.data))).collect(),
            dtype_policy: self.dtype_policy, missing: Vec::new(), errors: Vec::new(),
        };
        let module = module.map(&mut mapper);
        if validate && !mapper.errors.is_empty() {
            return Err(ModuleRecordError::Validation(format!("Apply errors: {:?}", mapper.errors)));
        }
        if !allow_partial && !mapper.missing.is_empty() {
            return Err(ModuleRecordError::Validation(format!("Missing tensors: {:?}", mapper.missing)));
        }
        if !allow_unused && !mapper.tensors.is_empty() {
            let mut unused: Vec<_> = mapper.tensors.keys().collect();
            unused.sort_unstable();
            return Err(ModuleRecordError::Validation(format!("Unused tensors: {unused:?}")));
        }
        Ok(module)
    }

    /// Serialize exact tensor storage, paths and parameter IDs with Rudapack.
    #[cfg(feature = "rudapack")]
    pub fn into_bytes(self) -> Result<ruda_tensor::api::Bytes, ModuleRecordError> {
        Ok(crate::RudapackWriter::new(self.into_snapshots()).to_bytes()?)
    }

    /// Load a backend-independent record from Rudapack bytes.
    #[cfg(feature = "rudapack")]
    pub fn from_bytes(bytes: ruda_tensor::api::Bytes) -> Result<Self, ModuleRecordError> {
        Self::from_reader(crate::rudapack::reader::RudapackReader::from_bytes(bytes)?)
    }

    /// Write the record without changing its tensor dtypes or parameter IDs.
    #[cfg(all(feature = "std", feature = "rudapack"))]
    pub fn save<P: AsRef<std::path::Path>>(self, path: P) -> Result<(), ModuleRecordError> {
        Ok(crate::RudapackWriter::new(self.into_snapshots()).write_to_file(path)?)
    }

    /// Read a Rudapack file without selecting a tensor backend or device.
    #[cfg(all(feature = "std", feature = "rudapack"))]
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self, ModuleRecordError> {
        Self::from_reader(crate::rudapack::reader::RudapackReader::from_file(path)?)
    }

    #[cfg(feature = "rudapack")]
    fn into_snapshots(self) -> Vec<crate::TensorSnapshot> {
        self.tensors.into_iter().map(|tensor| crate::TensorSnapshot::from_data(
            tensor.data, alloc::vec![tensor.path], Vec::new(), tensor.id)).collect()
    }

    #[cfg(feature = "rudapack")]
    fn from_reader(reader: crate::rudapack::reader::RudapackReader) -> Result<Self, ModuleRecordError> {
        let tensors = reader.get_snapshots_zero_copy(true)?.into_iter().map(|snapshot| {
            Ok(RecordTensor { path: snapshot.full_path(), id: snapshot.tensor_id.unwrap_or_else(ParamId::new),
                data: snapshot.to_data().map_err(|error| ModuleRecordError::Io(error.to_string()))? })
        }).collect::<Result<Vec<_>, ModuleRecordError>>()?;
        Ok(Self::from_tensors(tensors))
    }
}

struct RecordCollector {
    path: Vec<String>,
    group: Option<ParamGroup>,
    tensors: Vec<RecordTensor>,
}

impl RecordCollector {
    fn record(&mut self, path: String, id: ParamId, data: impl FnOnce() -> TensorData) {
        if self.group.as_ref().is_some_and(|group| !group.matches(&id, Some(&path))) { return; }
        self.tensors.push(RecordTensor { path, id, data: data() });
    }
}

macro_rules! visit_kind {
    ($method:ident, $path_method:ident, $kind:ty) => {
        fn $method<const D: usize>(&mut self, param: &Param<Tensor<B, D, $kind>>) {
            self.record(self.path.join("."), param.id, || param.transform_for_save().val().into_data());
        }
        fn $path_method<const D: usize>(&mut self, path: &[String], id: ParamId, tensor: &Tensor<B, D, $kind>) {
            self.record(path.join("."), id, || tensor.clone().into_data());
        }
    };
}

impl<B: Backend> ModuleVisitor<B> for RecordCollector {
    fn enter_module(&mut self, name: &str, _: &str) { self.path.push(name.to_string()); }
    fn exit_module(&mut self, _: &str, _: &str) { self.path.pop(); }
    visit_kind!(visit_float, visit_float_with_path, Float);
    visit_kind!(visit_int, visit_int_with_path, Int);
    visit_kind!(visit_bool, visit_bool_with_path, Bool);
}

struct RecordMapper {
    path: Vec<String>,
    tensors: HashMap<String, (ParamId, TensorData)>,
    dtype_policy: DTypePolicy,
    missing: Vec<String>,
    errors: Vec<String>,
}

impl RecordMapper {
    fn take<B: Backend, const D: usize, K: BasicOps<B>>(&mut self, device: &B::Device,
        shape: Shape, module_dtype: impl FnOnce() -> DType) -> Option<(Tensor<B, D, K>, ParamId)> {
        let path = self.path.join(".");
        let (id, data) = match self.tensors.remove(&path) {
            Some(value) => value,
            None => { self.missing.push(path); return None; }
        };
        let dtype = match self.dtype_policy {
            DTypePolicy::FromRecord => data.dtype,
            DTypePolicy::CastToModule => module_dtype(),
        };
        if data.shape != shape {
            self.errors.push(format!("{path}: shape mismatch, expected {shape:?} but record has {:?}", data.shape));
            return None;
        }
        Some((Tensor::from_data(data, (device, dtype)), id))
    }
}

macro_rules! map_kind {
    ($method:ident, $kind:ty) => {
        fn $method<const D: usize>(&mut self, param: Param<Tensor<B, D, $kind>>) -> Param<Tensor<B, D, $kind>> {
            let device = param.lazy_device();
            let shape = param.lazy_shape();
            match self.take(&device, shape, || param.val().dtype()) {
                Some((tensor, id)) => param.transform_for_load(tensor, id),
                None => param,
            }
        }
    };
}

impl<B: Backend> ModuleMapper<B> for RecordMapper {
    fn enter_module(&mut self, name: &str, _: &str) { self.path.push(name.to_string()); }
    fn exit_module(&mut self, _: &str, _: &str) { self.path.pop(); }
    map_kind!(map_float, Float);
    map_kind!(map_int, Int);
    map_kind!(map_bool, Bool);
}
