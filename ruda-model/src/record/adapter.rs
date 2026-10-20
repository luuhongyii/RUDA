use alloc::{string::{String, ToString}, vec, vec::Vec};
use hashbrown::HashMap;
use ruda_tensor::{Backend, DType, TensorData, api::{Bool, Float, Int, Tensor}};
use crate::module::{Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use super::{RuntimePrecision, RuntimeRecordValue as Value};

const PARAMETER_PATHS: u32 = 1;

pub(crate) fn capture<B: Backend, M: Module<B>>(module: &M, precision: Option<RuntimePrecision>) -> Value {
    let mut collector = Collector { path: Vec::new(), tensors: Vec::new(), precision };
    module.visit(&mut collector);
    Value::Fields(PARAMETER_PATHS, collector.tensors)
}

pub(crate) fn restore<B: Backend, M: Module<B>>(module: M, record: Value, backend_dtype: bool) -> M {
    let (tag, entries) = record.into_variant();
    assert_eq!(tag, PARAMETER_PATHS, "Invalid adapter parameter record format");
    let tensors = entries.into_iter().map(|entry| {
        let mut fields = entry.into_fields(0, 3);
        let path: String = fields.next().unwrap().into_scalar();
        let id: String = fields.next().unwrap().into_scalar();
        (path, (ParamId::deserialize(&id), fields.next().unwrap()))
    }).collect();
    let mut mapper = Mapper { path: Vec::new(), tensors, missing: Vec::new(), backend_dtype };
    let module = module.map(&mut mapper);
    assert!(mapper.missing.is_empty(), "Missing adapter parameters: {:?}", mapper.missing);
    if !mapper.tensors.is_empty() {
        let mut unused: Vec<_> = mapper.tensors.keys().collect();
        unused.sort_unstable();
        panic!("Unused adapter parameters: {unused:?}");
    }
    module
}

struct Collector {
    path: Vec<String>,
    tensors: Vec<Value>,
    precision: Option<RuntimePrecision>,
}

impl Collector {
    fn record(&mut self, path: String, id: ParamId, data: Value) {
        let data = match self.precision {
            Some(precision) => data.convert_precision(precision),
            None => data,
        };
        self.tensors.push(Value::Fields(0, vec![Value::scalar(path), Value::scalar(id.serialize()), data]));
    }
}

macro_rules! visit_kind {
    ($method:ident, $path_method:ident, $kind:ty, $variant:ident) => {
        fn $method<const D: usize>(&mut self, param: &Param<Tensor<B, D, $kind>>) {
            let data = param.transform_for_save().val().into_data();
            self.record(self.path.join("."), param.id, Value::$variant(data));
        }
        fn $path_method<const D: usize>(&mut self, path: &[String], id: ParamId, tensor: &Tensor<B, D, $kind>) {
            self.record(path.join("."), id, Value::$variant(tensor.clone().into_data()));
        }
    };
}

impl<B: Backend> ModuleVisitor<B> for Collector {
    fn enter_module(&mut self, name: &str, _: &str) { self.path.push(name.to_string()); }
    fn exit_module(&mut self, _: &str, _: &str) { self.path.pop(); }
    visit_kind!(visit_float, visit_float_with_path, Float, Float);
    visit_kind!(visit_int, visit_int_with_path, Int, Int);
    visit_kind!(visit_bool, visit_bool_with_path, Bool, Bool);
}

struct Mapper {
    path: Vec<String>,
    tensors: HashMap<String, (ParamId, Value)>,
    missing: Vec<String>,
    backend_dtype: bool,
}

impl Mapper {
    fn take(&mut self) -> Option<(ParamId, Value)> {
        let path = self.path.join(".");
        match self.tensors.remove(&path) {
            Some(value) => Some(value),
            None => { self.missing.push(path); None }
        }
    }
}

fn float_data<B: Backend>(data: TensorData, backend_dtype: bool) -> TensorData {
    if backend_dtype && !matches!(data.dtype, DType::QFloat(_)) { data.convert::<B::FloatElem>() } else { data }
}

fn int_data<B: Backend>(data: TensorData, backend_dtype: bool) -> TensorData {
    if backend_dtype { data.convert::<B::IntElem>() } else { data }
}

macro_rules! map_kind {
    ($method:ident, $kind:ty, $variant:ident, $convert:expr) => {
        fn $method<const D: usize>(&mut self, param: Param<Tensor<B, D, $kind>>) -> Param<Tensor<B, D, $kind>> {
            let Some((id, value)) = self.take() else { return param; };
            let (value, backend_dtype) = match value {
                Value::NativeFloat(data) => (Value::Float(data), false),
                Value::NativeInt(data) => (Value::Int(data), false),
                value => (value, self.backend_dtype),
            };
            let Value::$variant(data) = value else { panic!("Adapter parameter kind mismatch at {}", self.path.join(".")); };
            assert_eq!(data.shape, param.lazy_shape(), "Adapter parameter shape mismatch at {}", self.path.join("."));
            let data = ($convert)(data, backend_dtype);
            let dtype = data.dtype;
            let device = param.lazy_device();
            let tensor = B::memory_persistent_allocations(&device, data,
                |data| Tensor::<B, D, $kind>::from_data(data, (&device, dtype)));
            param.transform_for_load(tensor, id)
        }
    };
}

impl<B: Backend> ModuleMapper<B> for Mapper {
    fn enter_module(&mut self, name: &str, _: &str) { self.path.push(name.to_string()); }
    fn exit_module(&mut self, _: &str, _: &str) { self.path.pop(); }
    map_kind!(map_float, Float, Float, float_data::<B>);
    map_kind!(map_int, Int, Int, int_data::<B>);
    map_kind!(map_bool, Bool, Bool, |data: TensorData, _: bool| data);
}
