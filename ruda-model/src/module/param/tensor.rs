use super::{Param, ParamId, Parameter};
use crate::module::{
    AutodiffModule, Content, HasAutodiffModule, Module, ModuleDisplay, ModuleDisplayDefault,
    ModuleMapper, ModuleVisitor,
};
use crate::tensor::{
    Tensor,
    backend::{AutodiffBackend, Backend},
};
use alloc::{format, string::ToString, vec::Vec};
use ruda_tensor::api::{Bool, Float, Int, TensorData, ops::Device};

impl<B: Backend, const D: usize> Parameter for Tensor<B, D, Float> {
    type Device = B::Device;

    fn device(&self) -> Self::Device {
        Tensor::device(self)
    }

    fn is_require_grad(&self) -> bool {
        Tensor::is_require_grad(self)
    }

    fn set_require_grad(self, require_grad: bool) -> Self {
        if self.is_autodiff() { Tensor::set_require_grad(self, require_grad) } else { self }
    }

    fn into_materialized(self, require_grad: bool) -> Self {
        Parameter::set_require_grad(self.detach(), require_grad)
    }

    fn mapped_require_grad(&self, previous: bool) -> bool {
        if self.is_autodiff() { self.is_require_grad() } else { previous }
    }

    fn autodiff_context(&self) -> Option<ruda_tensor::AutodiffContext> { Some(Tensor::<B, D>::autodiff_context(self)) }
    fn creation_context(device: &Self::Device) -> Option<ruda_tensor::AutodiffContext> { Some(B::autodiff_creation_context(device)) }
    fn in_creation_context<R>(device: &Self::Device, context: Option<ruda_tensor::AutodiffContext>, func: impl FnOnce() -> R) -> R {
        match context { Some(context) => B::with_autodiff_creation_context(device, context, func), None => func() }
    }
}

impl<B: Backend, const D: usize> Parameter for Tensor<B, D, Int> {
    type Device = B::Device;

    fn device(&self) -> Self::Device {
        Tensor::device(self)
    }

    fn is_require_grad(&self) -> bool {
        false
    }

    fn set_require_grad(self, _require_grad: bool) -> Self {
        self
    }

    fn autodiff_context(&self) -> Option<ruda_tensor::AutodiffContext> { Some(Tensor::<B, D, Int>::autodiff_context(self)) }
    fn creation_context(device: &Self::Device) -> Option<ruda_tensor::AutodiffContext> { Some(B::autodiff_creation_context(device)) }
    fn in_creation_context<R>(device: &Self::Device, context: Option<ruda_tensor::AutodiffContext>, func: impl FnOnce() -> R) -> R {
        match context { Some(context) => B::with_autodiff_creation_context(device, context, func), None => func() }
    }
}

impl<B: Backend, const D: usize> Parameter for Tensor<B, D, Bool> {
    type Device = B::Device;

    fn device(&self) -> Self::Device {
        Tensor::device(self)
    }

    fn is_require_grad(&self) -> bool {
        false
    }

    fn set_require_grad(self, _require_grad: bool) -> Self {
        self
    }

    fn autodiff_context(&self) -> Option<ruda_tensor::AutodiffContext> { Some(Tensor::<B, D, Bool>::autodiff_context(self)) }
    fn creation_context(device: &Self::Device) -> Option<ruda_tensor::AutodiffContext> { Some(B::autodiff_creation_context(device)) }
    fn in_creation_context<R>(device: &Self::Device, context: Option<ruda_tensor::AutodiffContext>, func: impl FnOnce() -> R) -> R {
        match context { Some(context) => B::with_autodiff_creation_context(device, context, func), None => func() }
    }
}

impl<B: Backend, const D: usize> Param<Tensor<B, D>> {
    /// Create a new parameter from a float tensor.
    ///
    /// # Warnings
    ///
    /// We strongly recommend using [Param::uninitialized] if you are using this method to
    /// initialize parameters inside a module, since the tensor initialization will be lazy,
    /// making the loading of weights more performant.
    pub fn from_tensor(value: Tensor<B, D>) -> Self {
        // When creating a parameter from a float tensor, we automatically mark it as requiring
        // gradients, so that it can be updated by an optimizer.
        let mut param = Param::initialized(ParamId::new(), Parameter::set_require_grad(value, true));
        param.require_grad = true;
        param
    }

    /// The shape of the parameter, **without triggering initialization**.
    ///
    /// This is critical for shape validation during loading: when applying tensors to an
    /// uninitialized parameter, we need to validate the shape without triggering the
    /// initialization function (which would allocate an unnecessary tensor).
    ///
    /// Use this instead of [crate::tensor::Tensor::shape] when you need the shape but want to
    /// preserve lazy initialization.
    pub fn lazy_shape(&self) -> ruda_tensor::api::Shape {
        let initialization = match &self.initialization {
            Some(init) => init,
            None => return self.shape(),
        };

        let init = initialization.read().unwrap();

        match init.as_ref() {
            Some(value) => value.shape.clone(),
            None => self.shape(),
        }
    }

    /// Create a new parameter from data.
    pub fn from_data<T>(data: T, device: &B::Device) -> Self
    where
        T: Into<TensorData>,
    {
        let data: TensorData = data.into();
        // When creating a parameter from a float tensor, we automatically mark it as requiring
        // gradients, so that it can be updated by an optimizer.
        B::memory_persistent_allocations(device, data, |data| {
            let value = Tensor::from_data(data, device);
            Self::from_tensor(value)
        })
    }

    /// Transform a parameter for loading by applying load transformations.
    ///
    /// This method is used to restore a parameter from a tensor (typically during deserialization).
    /// It ensures the tensor is moved to the expected device, applies the param mapper's
    /// `on_load` transformation, and preserves the autodiff settings (require_grad).
    pub fn transform_for_load(self, tensor: Tensor<B, D>, param_id: ParamId) -> Self {
        let mut new_tensor = tensor;
        let context = self.lazy_autodiff_context();

        let mapper = self.param_mapper.clone();

        let expected_device = self.lazy_device();
        let expected_require_grad = self.lazy_is_require_grad();

        // Make sure we load the tensor into the same module device.
        if new_tensor.device() != expected_device {
            new_tensor = new_tensor.to_device(&expected_device).detach();
        }

        new_tensor = mapper.on_load(new_tensor);

        // Make sure we load the tensor with the same autodiff setting.
        if let Some(context) = context { new_tensor = new_tensor.with_autodiff_context(context); }
        new_tensor = Parameter::set_require_grad(new_tensor, expected_require_grad);

        let mut loaded = Self::initialized(param_id, new_tensor);
        loaded.param_mapper = mapper;
        loaded.require_grad = self.require_grad;
        loaded.transform = self.transform;
        loaded
    }

    /// Transform a parameter for saving by applying save transformations.
    ///
    /// This method is used to prepare a parameter for saving (typically during serialization).
    /// It applies the param mapper's `on_save` transformation, which can be used
    /// to modify the tensor before serialization (e.g., quantization, precision conversion).
    pub fn transform_for_save(&self) -> Self {
        let mut tensor = self.base();
        let mapper = self.param_mapper.clone();

        tensor = mapper.on_save(tensor);

        let mut record = Self::initialized(self.id, tensor);
        record.require_grad = self.require_grad;
        record.transform = self.transform.clone();
        record.transform_record = self.transform_record.clone();
        record
    }
}

impl<B: Backend, const D: usize> Param<Tensor<B, D, Int>> {
    /// The shape of the parameter, **without triggering initialization**.
    ///
    /// This is critical for shape validation during loading: when applying tensors to an
    /// uninitialized parameter, we need to validate the shape without triggering the
    /// initialization function (which would allocate an unnecessary tensor).
    ///
    /// Use this instead of [crate::tensor::Tensor::shape] when you need the shape but want to
    /// preserve lazy initialization.
    pub fn lazy_shape(&self) -> ruda_tensor::api::Shape {
        let initialization = match &self.initialization {
            Some(init) => init,
            None => return self.shape(),
        };

        let init = initialization.read().unwrap();

        match init.as_ref() {
            Some(value) => value.shape.clone(),
            None => self.shape(),
        }
    }

    /// Transform a parameter for loading by applying load transformations.
    ///
    /// This method is used to restore a parameter from a tensor (typically during deserialization).
    /// It ensures the tensor is moved to the expected device and applies the param mapper's
    /// `on_load` transformation.
    pub fn transform_for_load(self, tensor: Tensor<B, D, Int>, param_id: ParamId) -> Self {
        let mut new_tensor = tensor;
        let context = self.lazy_autodiff_context();

        let mapper = self.param_mapper.clone();

        let expected_device = self.lazy_device();

        // Make sure we load the tensor into the same module device.
        if new_tensor.device() != expected_device {
            new_tensor = new_tensor.to_device(&expected_device);
        }

        new_tensor = mapper.on_load(new_tensor);
        if let Some(context) = context { new_tensor = new_tensor.with_autodiff_context(context); }

        let mut loaded = Self::initialized(param_id, new_tensor);
        loaded.param_mapper = mapper;
        loaded
    }

    /// Transform a parameter for saving by applying save transformations.
    ///
    /// This method is used to prepare a parameter for saving (typically during serialization).
    /// It applies the param mapper's `on_save` transformation, which can be used
    /// to modify the tensor before serialization (e.g., quantization, precision conversion).
    pub fn transform_for_save(&self) -> Self {
        let mut tensor = self.val();
        let mapper = self.param_mapper.clone();

        tensor = mapper.on_save(tensor);

        Self::initialized(self.id, tensor)
    }
}

impl<B: Backend, const D: usize> Param<Tensor<B, D, Bool>> {
    /// The shape of the parameter, **without triggering initialization**.
    ///
    /// This is critical for shape validation during loading: when applying tensors to an
    /// uninitialized parameter, we need to validate the shape without triggering the
    /// initialization function (which would allocate an unnecessary tensor).
    ///
    /// **Returns:**
    /// - For uninitialized params: the shape from the `Uninitialized` struct
    /// - For initialized params: the actual shape from the tensor
    ///
    /// Use this instead of [crate::tensor::Tensor::shape] when you need the shape but want to
    /// preserve lazy initialization.
    pub fn lazy_shape(&self) -> ruda_tensor::api::Shape {
        let initialization = match &self.initialization {
            Some(init) => init,
            None => return self.shape(),
        };

        let init = initialization.read().unwrap();

        match init.as_ref() {
            Some(value) => value.shape.clone(),
            None => self.shape(),
        }
    }

    /// Transform a parameter for loading by applying load transformations.
    ///
    /// This method is used to restore a parameter from a tensor (typically during deserialization).
    /// It ensures the tensor is moved to the expected device and applies the param mapper's
    /// `on_load` transformation.
    pub fn transform_for_load(self, tensor: Tensor<B, D, Bool>, param_id: ParamId) -> Self {
        let mut new_tensor = tensor;
        let context = self.lazy_autodiff_context();

        let mapper = self.param_mapper.clone();

        let expected_device = self.lazy_device();

        // Make sure we load the tensor into the same module device.
        if new_tensor.device() != expected_device {
            new_tensor = new_tensor.to_device(&expected_device);
        }

        new_tensor = mapper.on_load(new_tensor);
        if let Some(context) = context { new_tensor = new_tensor.with_autodiff_context(context); }

        let mut loaded = Self::initialized(param_id, new_tensor);
        loaded.param_mapper = mapper;
        loaded
    }

    /// Transform a parameter for saving by applying save transformations.
    ///
    /// This method is used to prepare a parameter for saving (typically during serialization).
    /// It applies the param mapper's `on_save` transformation, which can be used
    /// to modify the tensor before serialization (e.g., quantization, precision conversion).
    pub fn transform_for_save(&self) -> Self {
        let mut tensor = self.val();
        let mapper = self.param_mapper.clone();

        tensor = mapper.on_save(tensor);

        Self::initialized(self.id, tensor)
    }
}

impl<const D: usize, B: Backend> Module<B> for Param<Tensor<B, D>> {
    type Record = Param<Tensor<B, D>>;

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        if visitor.handles_reparameterizations() { visitor.visit_float(self); return; }
        match &self.transform {
            None => visitor.visit_float(self),
            Some(transform) => {
                visitor.visit_float(&self.clone().without_transform());
                visitor.enter_module(transform.name(), "Reparameterization");
                super::transform_traversal::visit::<B, _, _>(transform.as_ref(), visitor);
                visitor.exit_module(transform.name(), "Reparameterization");
            }
        }
    }

    fn map<M: ModuleMapper<B>>(mut self, mapper: &mut M) -> Self {
        if mapper.handles_reparameterizations() { return mapper.map_float(self); }
        if self.transform.is_some() && mapper.materialize_reparameterizations() {
            return mapper.map_float(self.materialize());
        }
        let Some(transform) = self.transform.take() else { return mapper.map_float(self); };
        assert!(!mapper.attaches_reparameterizations(), "Nested reparameterizations are not supported");
        let mut base = mapper.map_float(self);
        if base.transform.is_some() { return base; }
        mapper.enter_module(transform.name(), "Reparameterization");
        base.transform = Some(super::transform_traversal::map::<B, _, _>(transform.as_ref(), mapper));
        mapper.exit_module(transform.name(), "Reparameterization");
        base.transform_record = None;
        base
    }

    fn into_record(self) -> Self::Record {
        self.transform_for_save()
    }

    fn load_record(self, record: Self::Record) -> Self {
        assert!((record.transform_record.is_none() && record.transform.is_none()) || self.transform.is_some(),
            "Record contains a reparameterization absent from the target parameter");
        let transform = match (&self.transform, &record.transform) {
            (Some(target), Some(source)) => Some(target.load(source.as_any())),
            (Some(target), None) => match &record.transform_record {
                Some(record) => Some(target.load_record(record, &self.lazy_device())),
                None => Some(target.clone()),
            },
            (target, _) => target.clone(),
        };
        let (record_param_id, record_tensor, _) = record.consume();
        let mut loaded = self.transform_for_load(record_tensor, record_param_id);
        loaded.transform = transform;
        loaded.transform_record = None;
        loaded
    }

    fn to_device(self, device: &Device<B>) -> Self {
        let require_grad = self.require_grad;
        let transform = self.transform.as_ref().map(|state| state.to_device(device));
        let mut param = self.map(|tensor| tensor.to_device(device));
        param.require_grad = require_grad;
        param.transform = transform;
        param
    }

    fn fork(self, device: &Device<B>) -> Self {
        let require_grad = self.require_grad;
        let transform = self.transform.as_ref().map(|state| state.fork(device));
        let mut param = self.map(|tensor| {
            let is_require_grad = tensor.is_require_grad();
            let mut tensor = tensor.to_device(device).detach();

            if is_require_grad {
                tensor = tensor.require_grad();
            }

            tensor
        });
        param.require_grad = require_grad;
        param.transform = transform;
        param
    }

    fn collect_devices(&self, mut devices: Vec<Device<B>>) -> Vec<Device<B>> {
        let device = self.lazy_device();

        if !devices.contains(&device) {
            devices.push(device)
        }

        match &self.transform {
            Some(state) => state.collect_devices(devices),
            None => devices,
        }
    }
}

impl<const D: usize, B: Backend> ModuleDisplayDefault for Param<Tensor<B, D>> {
    fn content(&self, content: Content) -> Option<Content> {
        let id = if content.display_settings.show_param_id() {
            format!(", id: {}", self.id)
        } else {
            "".to_string()
        };
        let string = format!(
            "ParamTensor {{rank: {D}, shape: {:?}, kind: float{id}}}",
            self.shape().as_slice()
        );
        content.add_formatted(&string).optional()
    }
}
impl<const D: usize, B: Backend> ModuleDisplay for Param<Tensor<B, D>> {}

impl<const D: usize, B: Backend> Module<B> for Param<Tensor<B, D, Int>> {
    type Record = Param<Tensor<B, D, Int>>;

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        visitor.visit_int(self)
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        mapper.map_int(self)
    }

    fn into_record(self) -> Self::Record {
        self.transform_for_save()
    }

    fn load_record(self, record: Self::Record) -> Self {
        let (record_param_id, record_tensor, _) = record.consume();
        self.transform_for_load(record_tensor, record_param_id)
    }

    fn to_device(self, device: &Device<B>) -> Self {
        self.map(|tensor| tensor.to_device(device))
    }

    fn fork(self, device: &Device<B>) -> Self {
        self.to_device(device) // Don't support autodiff.
    }

    fn collect_devices(&self, mut devices: Vec<Device<B>>) -> Vec<Device<B>> {
        let device = self.val().device();

        if !devices.contains(&device) {
            devices.push(device)
        }

        devices
    }
}

impl<const D: usize, B: Backend> ModuleDisplayDefault for Param<Tensor<B, D, Int>> {
    fn content(&self, content: Content) -> Option<Content> {
        let id = if content.display_settings.show_param_id() {
            format!(", id: {}", self.id)
        } else {
            "".to_string()
        };
        let string = format!(
            "ParamTensor {{rank: {D}, shape: {:?}, kind: int{id}}}",
            self.shape().as_slice()
        );
        content.add_formatted(&string).optional()
    }
}
impl<const D: usize, B: Backend> ModuleDisplay for Param<Tensor<B, D, Int>> {}

impl<const D: usize, B: Backend> Module<B> for Param<Tensor<B, D, Bool>> {
    type Record = Param<Tensor<B, D, Bool>>;

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        visitor.visit_bool(self)
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        mapper.map_bool(self)
    }

    fn into_record(self) -> Self::Record {
        self.transform_for_save()
    }

    fn load_record(self, record: Self::Record) -> Self {
        let (record_param_id, record_tensor, _) = record.consume();
        self.transform_for_load(record_tensor, record_param_id)
    }

    fn to_device(self, device: &Device<B>) -> Self {
        self.map(|tensor| tensor.to_device(device))
    }

    fn fork(self, device: &Device<B>) -> Self {
        self.to_device(device) // Don't support autodiff.
    }

    fn collect_devices(&self, mut devices: Vec<Device<B>>) -> Vec<Device<B>> {
        let device = self.val().device();

        if !devices.contains(&device) {
            devices.push(device)
        }

        devices
    }
}

impl<const D: usize, B: Backend> ModuleDisplayDefault for Param<Tensor<B, D, Bool>> {
    fn content(&self, content: Content) -> Option<Content> {
        let id = if content.display_settings.show_param_id() {
            format!(", id: {}", self.id)
        } else {
            "".to_string()
        };

        let string = format!(
            "ParamTensor {{rank: {D}, shape: {:?}, kind: bool{id}}}",
            self.shape().as_slice()
        );
        content.add_formatted(&string).optional()
    }
}

impl<const D: usize, B: Backend> ModuleDisplay for Param<Tensor<B, D, Bool>> {}

impl<const D: usize, B: AutodiffBackend> AutodiffModule<B> for Param<Tensor<B, D>> {
    type InnerModule = Param<Tensor<B::InnerBackend, D>>;

    fn valid(&self) -> Self::InnerModule {
        // Preserve initialized param `require_grad` state, but reset the inner value's
        let require_grad = self.require_grad;
        let mut param = Param::initialized(self.id, self.base().inner().set_require_grad(false));
        param.require_grad = require_grad;
        param.param_mapper = self.param_mapper.clone().convert(
            Tensor::<B, D>::from_inner, Tensor::<B, D>::inner,
        );
        param.transform_record = self.transform_record.clone();
        param
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        // Reinstate the param's `require_grad` state
        let tensor = Tensor::from_inner(module.base()).set_require_grad(module.require_grad);
        let mut param = Param::initialized(module.id, tensor);
        param.require_grad = module.require_grad;
        param.param_mapper = module.param_mapper.convert(
            Tensor::<B, D>::inner, Tensor::<B, D>::from_inner,
        );
        param.transform_record = module.transform_record;
        param
    }
}

impl<const D: usize, B: AutodiffBackend> HasAutodiffModule<B>
    for Param<Tensor<B::InnerBackend, D>>
{
    type TrainModule = Param<Tensor<B, D>>;
}

impl<const D: usize, B: AutodiffBackend> AutodiffModule<B> for Param<Tensor<B, D, Int>> {
    type InnerModule = Param<Tensor<B::InnerBackend, D, Int>>;

    fn valid(&self) -> Self::InnerModule {
        let mut param = Param::initialized(self.id, self.base().inner());
        param.require_grad = self.require_grad;
        param.param_mapper = self.param_mapper.clone().convert(
            Tensor::<B, D, Int>::from_inner, Tensor::<B, D, Int>::inner,
        );
        param.transform_record = self.transform_record.clone();
        param
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        let mut param = Param::initialized(module.id, Tensor::from_inner(module.base()));
        param.require_grad = module.require_grad;
        param.param_mapper = module.param_mapper.convert(
            Tensor::<B, D, Int>::inner, Tensor::<B, D, Int>::from_inner,
        );
        param.transform_record = module.transform_record;
        param
    }
}

impl<const D: usize, B: AutodiffBackend> AutodiffModule<B> for Param<Tensor<B, D, Bool>> {
    type InnerModule = Param<Tensor<B::InnerBackend, D, Bool>>;

    fn valid(&self) -> Self::InnerModule {
        let mut param = Param::initialized(self.id, self.base().inner());
        param.require_grad = self.require_grad;
        param.param_mapper = self.param_mapper.clone().convert(
            Tensor::<B, D, Bool>::from_inner, Tensor::<B, D, Bool>::inner,
        );
        param.transform_record = self.transform_record.clone();
        param
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        let mut param = Param::initialized(module.id, Tensor::from_inner(module.base()));
        param.require_grad = module.require_grad;
        param.param_mapper = module.param_mapper.convert(
            Tensor::<B, D, Bool>::inner, Tensor::<B, D, Bool>::from_inner,
        );
        param.transform_record = module.transform_record;
        param
    }
}

impl<const D: usize, B: AutodiffBackend> HasAutodiffModule<B>
    for Param<Tensor<B::InnerBackend, D, Int>>
{
    type TrainModule = Param<Tensor<B, D, Int>>;
}

impl<const D: usize, B: AutodiffBackend> HasAutodiffModule<B>
    for Param<Tensor<B::InnerBackend, D, Bool>>
{
    type TrainModule = Param<Tensor<B, D, Bool>>;
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::{
        TestAutodiffBackend,
        module::Module,
        record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
    };

    #[test]
    fn test_load_record_setting() {
        let device = Default::default();
        let tensor = Tensor::<TestAutodiffBackend, 2>::ones([3, 3], &device).require_grad();

        let byte_recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let bytes = byte_recorder
            .record(
                Param::initialized(ParamId::new(), tensor.clone()).into_record(),
                (),
            )
            .unwrap();

        let no_grad_is_require_grad = Param::initialized(ParamId::new(), tensor.clone())
            .no_grad()
            .load_record(byte_recorder.load(bytes.clone(), &device).unwrap())
            .is_require_grad();

        let with_default_is_require_grad = Param::initialized(ParamId::new(), tensor)
            .load_record(byte_recorder.load(bytes, &device).unwrap())
            .is_require_grad();

        assert!(!no_grad_is_require_grad);
        assert!(with_default_is_require_grad);
    }

    #[test]
    fn test_param_require_grad_stateful() {
        let device = Default::default();
        let tensor = Tensor::<TestAutodiffBackend, 2>::ones([3, 3], &device).require_grad();

        let param = Param::initialized(ParamId::new(), tensor);
        assert!(param.is_require_grad());
        assert!(param.require_grad);

        let param = param.valid();
        assert!(!param.is_require_grad());
        assert!(param.require_grad); // stateful

        // Without `HasAutodiffModule`, we would need to specify the param type as well, which would be annoying:
        // let param: Param<Tensor<TestAutodiffBackend, _>> = param.train();
        let param = param.train::<TestAutodiffBackend>();
        assert!(param.is_require_grad());
        assert!(param.require_grad); // stateful

        let param = param.no_grad();
        assert!(!param.is_require_grad());
        assert!(!param.require_grad); // stateful

        let param = param.valid();
        assert!(!param.is_require_grad()); // always
        assert!(!param.require_grad); // stateful

        let param = param.train::<TestAutodiffBackend>();
        assert!(!param.is_require_grad());
        assert!(!param.require_grad); // stateful
    }
}
