use super::{AutodiffModule, Content, Devices, HasAutodiffModule, Module, ModuleDisplay,
    ModuleDisplayDefault, ModuleMapper, ModuleVisitor, Param, Flag};
use super::param::transform::{self, ParamTransform, TransformRecord};
use crate::record::{PrecisionSettings, Record, RuntimePrecision, RuntimeRecord};
use alloc::{string::{String, ToString}, vec::Vec};
use core::{any::Any, marker::PhantomData, ops::Deref};
use ruda_tensor::{Backend, AutodiffBackend, Tensor, Int, Bool};

/// A module whose state transforms a stored floating-point parameter.
pub trait Reparameterization<B: Backend>: Module<B> + Sync + 'static {
    /// Path component for the nested parameters.
    const NAME: &'static str;
    /// Materialize an effective value without replacing the stored base.
    fn apply<const D: usize>(&self, base: Tensor<B, D>) -> Tensor<B, D>;
}

/// Select and prepare structural parameters and their transformations.
pub trait Reparameterizer<B: Backend> {
    /// The concrete transformation module.
    type Reparam: Reparameterization<B>;
    /// Return the structural base and an optional attached transformation.
    fn reparameterize<const D: usize>(&mut self, path: &str, param: Param<Tensor<B, D>>)
        -> (Param<Tensor<B, D>>, Option<Self::Reparam>);
}

#[derive(Debug, Clone)]
struct Attachment<R> {
    path: String,
    state: R,
}

/// A model retaining the concrete adapter type through backend conversions.
/// Forward methods of the original model remain available through `Deref`.
#[derive(Debug)]
pub struct Reparameterized<M, R, B: Backend> {
    module: M,
    attachments: Vec<Attachment<R>>,
    owner: usize,
    backend: PhantomData<fn() -> B>,
}

impl<M: Clone, R: Clone, B: Backend> Clone for Reparameterized<M, R, B> {
    fn clone(&self) -> Self {
        Self { module: self.module.clone(), attachments: self.attachments.clone(),
            owner: self.owner, backend: PhantomData }
    }
}

impl<M, R, B: Backend> Deref for Reparameterized<M, R, B> {
    type Target = M;
    fn deref(&self) -> &M { &self.module }
}

#[derive(Debug)]
struct Applied<R, B: Backend, const D: usize> {
    state: R,
    owner: usize,
    backend: PhantomData<fn() -> B>,
}

impl<R: Reparameterization<B>, B: Backend, const D: usize> ParamTransform<Tensor<B, D>> for Applied<R, B, D> {
    fn name(&self) -> &'static str { R::NAME }
    fn visit_dyn(&self, visitor: &mut dyn super::param::transform_traversal::DynVisitor) {
        self.state.visit(&mut super::param::transform_traversal::VisitorToDyn(visitor));
    }
    fn map_dyn(&self, mapper: &mut dyn super::param::transform_traversal::DynMapper) -> transform::SharedTransform<Tensor<B, D>> {
        transform::share(Self { state: self.state.clone().map(&mut super::param::transform_traversal::MapperToDyn(mapper)),
            owner: self.owner, backend: PhantomData })
    }
    fn apply(&self, base: Tensor<B, D>) -> Tensor<B, D> { self.state.apply(base) }
    fn as_any(&self) -> &dyn Any { &self.state }
    fn owner(&self) -> usize { self.owner }
    fn to_device(&self, device: &B::Device) -> transform::SharedTransform<Tensor<B, D>> {
        transform::share(Self { state: self.state.clone().to_device(device), owner: self.owner, backend: PhantomData })
    }
    fn fork(&self, device: &B::Device) -> transform::SharedTransform<Tensor<B, D>> {
        transform::share(Self { state: self.state.clone().fork(device), owner: self.owner, backend: PhantomData })
    }
    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> { self.state.collect_devices(devices) }
    fn load(&self, state: &dyn Any) -> transform::SharedTransform<Tensor<B, D>> {
        let source = state.downcast_ref::<R>().expect("Reparameterization record type mismatch");
        let record = crate::record::adapter::capture(source, None);
        transform::share(Self { state: crate::record::adapter::restore(self.state.clone(), record, false),
            owner: self.owner, backend: PhantomData })
    }
    fn record(&self, precision: RuntimePrecision) -> TransformRecord {
        TransformRecord::from_runtime(R::NAME.to_string(), crate::record::adapter::capture(&self.state, Some(precision)), Some(precision.id))
    }
    fn load_record(&self, record: &TransformRecord, _device: &B::Device) -> transform::SharedTransform<Tensor<B, D>> {
        assert_eq!(record.name, R::NAME, "Reparameterization record type mismatch");
        let state = crate::record::adapter::restore(self.state.clone(), record.runtime_value(), true);
        transform::share(Self { state, owner: self.owner, backend: PhantomData })
    }
    fn visit_float_primitives(&self, visitor: &mut dyn FnMut(super::ParamId, &dyn Any)) {
        struct Primitives<'a>(&'a mut dyn FnMut(super::ParamId, &dyn Any));
        impl<B: Backend> ModuleVisitor<B> for Primitives<'_> {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                (self.0)(param.id, &param.base().into_primitive());
                param.visit_reparameterization_primitives(self.0);
            }
        }
        self.state.visit(&mut Primitives(visitor));
    }
}

fn attach<B: Backend, R: Reparameterization<B>, const D: usize>(mut param: Param<Tensor<B, D>>, state: &R, owner: usize)
    -> Param<Tensor<B, D>>
{
    assert!(param.transform_owner().is_none_or(|existing| existing == owner),
        "Nested reparameterizations are not supported");
    param.transform = Some(transform::share(Applied::<R, B, D> { state: state.clone(), owner, backend: PhantomData }));
    param
}

fn next_owner() -> usize {
    #[cfg(target_has_atomic = "ptr")]
    use core::sync::atomic::{AtomicUsize, Ordering};
    #[cfg(not(target_has_atomic = "ptr"))]
    use portable_atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(1);
    NEXT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| value.checked_add(1))
        .expect("Reparameterization owner ID overflow")
}

pub(crate) fn apply<B: Backend, M: Module<B>, R: Reparameterizer<B>>(module: M, reparameterizer: R)
    -> Reparameterized<M, R::Reparam, B>
{
    struct Apply<R, A> { reparameterizer: R, attachments: Vec<Attachment<A>>, path: Vec<String>, owner: usize }
    impl<B: Backend, R: Reparameterizer<B>> ModuleMapper<B> for Apply<R, R::Reparam> {
        fn attaches_reparameterizations(&self) -> bool { true }
        fn enter_module(&mut self, name: &str, _: &str) { self.path.push(name.to_string()); }
        fn exit_module(&mut self, _: &str, _: &str) { self.path.pop(); }
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
            let path = self.path.join(".");
            let (base, state) = self.reparameterizer.reparameterize(&path, param);
            match state {
                Some(state) => {
                    let param = attach(base, &state, self.owner);
                    self.attachments.push(Attachment { path, state });
                    param
                }
                None => base,
            }
        }
    }
    let mut mapper = Apply { reparameterizer, attachments: Vec::new(), path: Vec::new(), owner: next_owner() };
    let module = module.map(&mut mapper);
    Reparameterized { module, attachments: mapper.attachments, owner: mapper.owner, backend: PhantomData }
}

struct Bind<'a, R> { attachments: &'a [Attachment<R>], path: Vec<String>, owner: usize }
impl<B: Backend, R: Reparameterization<B>> ModuleMapper<B> for Bind<'_, R> {
    fn handles_reparameterizations(&self) -> bool { true }
    fn enter_module(&mut self, name: &str, _: &str) { self.path.push(name.to_string()); }
    fn exit_module(&mut self, _: &str, _: &str) { self.path.pop(); }
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let path = self.path.join(".");
        match self.attachments.iter().find(|attachment| attachment.path == path) {
            Some(attachment) => attach(param, &attachment.state, self.owner),
            None => param,
        }
    }
}

impl<M: Module<B>, R: Reparameterization<B>, B: Backend> Reparameterized<M, R, B> {
    /// Merge all attached state and recover the original model type.
    pub fn into_materialized(self) -> M { self.materialize().module }

    #[doc(hidden)]
    pub fn with_module(mut self, module: M) -> Self {
        self.module = module;
        self.refresh()
    }

    fn refresh(mut self) -> Self {
        struct Refresh<R> { attachments: Vec<Attachment<R>>, path: Vec<String>, owner: usize }
        impl<B: Backend, R: Reparameterization<B>> ModuleVisitor<B> for Refresh<R> {
            fn handles_reparameterizations(&self) -> bool { true }
            fn enter_module(&mut self, name: &str, _: &str) { self.path.push(name.to_string()); }
            fn exit_module(&mut self, _: &str, _: &str) { self.path.pop(); }
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                if param.transform_owner() != Some(self.owner) { return; }
                let path = self.path.join(".");
                let state = param.reparameterization::<R>()
                    .expect("Reparameterization type changed while replacing a model").clone();
                self.attachments.push(Attachment { path, state });
            }
        }
        let mut visitor = Refresh { attachments: Vec::new(), path: Vec::new(), owner: self.owner };
        self.module.visit(&mut visitor);
        self.attachments = visitor.attachments;
        self.bind()
    }

    fn bind(mut self) -> Self {
        self.module = self.module.map(&mut Bind { attachments: &self.attachments, path: Vec::new(), owner: self.owner });
        self
    }
}

struct Visit<'a, R, V> {
    attachments: &'a [Attachment<R>], visitor: &'a mut V, path: Vec<String>, owner: usize,
}
impl<B: Backend, R: Reparameterization<B>, V: ModuleVisitor<B>> ModuleVisitor<B> for Visit<'_, R, V> {
    fn handles_reparameterizations(&self) -> bool { true }
    fn enter_module(&mut self, name: &str, kind: &str) {
        self.path.push(name.to_string()); self.visitor.enter_module(name, kind);
    }
    fn exit_module(&mut self, name: &str, kind: &str) {
        self.visitor.exit_module(name, kind); self.path.pop();
    }
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if param.transform_owner() != Some(self.owner) { self.visitor.visit_float(param); return; }
        self.visitor.visit_float(&param.clone().without_transform());
        let path = self.path.join(".");
        let state = &self.attachments.iter().find(|attachment| attachment.path == path)
            .expect("Attached parameter has no reparameterization state").state;
        self.visitor.enter_module(R::NAME, "Reparameterization");
        state.visit(self.visitor);
        self.visitor.exit_module(R::NAME, "Reparameterization");
    }
    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<B, D, Int>>) { self.visitor.visit_int(param); }
    fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<B, D, Bool>>) { self.visitor.visit_bool(param); }
    fn visit_flag(&mut self, param: &Param<Flag>) { self.visitor.visit_flag(param); }
    fn visit_float_with_path<const D: usize>(&mut self, path: &[String], id: super::ParamId, tensor: &Tensor<B, D>) {
        self.visitor.visit_float_with_path(path, id, tensor);
    }
    fn visit_int_with_path<const D: usize>(&mut self, path: &[String], id: super::ParamId, tensor: &Tensor<B, D, Int>) {
        self.visitor.visit_int_with_path(path, id, tensor);
    }
    fn visit_bool_with_path<const D: usize>(&mut self, path: &[String], id: super::ParamId, tensor: &Tensor<B, D, Bool>) {
        self.visitor.visit_bool_with_path(path, id, tensor);
    }
}

struct Map<'a, R, M> {
    attachments: &'a mut Vec<Attachment<R>>, mapper: &'a mut M, path: Vec<String>, owner: usize,
}
impl<B: Backend, R: Reparameterization<B>, M: ModuleMapper<B>> ModuleMapper<B> for Map<'_, R, M> {
    fn handles_reparameterizations(&self) -> bool { true }
    fn attaches_reparameterizations(&self) -> bool { self.mapper.attaches_reparameterizations() }
    fn map_tensor<const D: usize, K: ruda_tensor::BasicOps<B>>(&mut self, tensor: Tensor<B, D, K>) -> Tensor<B, D, K> {
        self.mapper.map_tensor(tensor)
    }
    fn materialize_reparameterizations(&self) -> bool { self.mapper.materialize_reparameterizations() }
    fn enter_module(&mut self, name: &str, kind: &str) {
        self.path.push(name.to_string()); self.mapper.enter_module(name, kind);
    }
    fn exit_module(&mut self, name: &str, kind: &str) {
        self.mapper.exit_module(name, kind); self.path.pop();
    }
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        if param.transform_owner() != Some(self.owner) { return self.mapper.map_float(param); }
        assert!(!self.mapper.attaches_reparameterizations(), "Nested reparameterizations are not supported");
        let path = self.path.join(".");
        let index = self.attachments.iter().position(|attachment| attachment.path == path)
            .expect("Attached parameter has no reparameterization state");
        if self.mapper.materialize_reparameterizations() {
            self.attachments.remove(index);
            return self.mapper.map_float(param.materialize());
        }
        let base = self.mapper.map_float(param.without_transform());
        if base.transform_owner().is_some_and(|owner| owner != self.owner) {
            self.attachments.remove(index);
            return base;
        }
        self.mapper.enter_module(R::NAME, "Reparameterization");
        let state = self.attachments[index].state.clone().map(self.mapper);
        self.mapper.exit_module(R::NAME, "Reparameterization");
        self.attachments[index].state = state;
        attach(base, &self.attachments[index].state, self.owner)
    }
    fn map_int<const D: usize>(&mut self, param: Param<Tensor<B, D, Int>>) -> Param<Tensor<B, D, Int>> { self.mapper.map_int(param) }
    fn map_bool<const D: usize>(&mut self, param: Param<Tensor<B, D, Bool>>) -> Param<Tensor<B, D, Bool>> { self.mapper.map_bool(param) }
    fn map_flag(&mut self, param: Param<Flag>) -> Param<Flag> { self.mapper.map_flag(param) }
}

impl<B: Backend, M: Module<B>, R: Reparameterization<B>> Module<B> for Reparameterized<M, R, B> {
    type Record = ReparameterizedRecord<M, R, B>;
    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        self.module.visit(&mut Visit { attachments: &self.attachments, visitor, path: Vec::new(), owner: self.owner });
    }
    fn map<V: ModuleMapper<B>>(mut self, mapper: &mut V) -> Self {
        self.module = self.module.map(&mut Map { attachments: &mut self.attachments, mapper, path: Vec::new(), owner: self.owner });
        self
    }
    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        self.attachments.iter().fold(self.module.collect_devices(devices), |devices, attachment| attachment.state.collect_devices(devices))
    }
    fn to_device(mut self, device: &B::Device) -> Self {
        self.module = self.module.to_device(device);
        self.refresh()
    }
    fn fork(mut self, device: &B::Device) -> Self {
        self.module = self.module.fork(device);
        self.refresh()
    }
    fn into_record(self) -> Self::Record {
        ReparameterizedRecord { state: RecordState::Live(self.module, self.attachments, self.owner) }
    }
    fn load_record(mut self, record: Self::Record) -> Self {
        match record.state {
            RecordState::Live(module, _, _) => {
                self.module = self.module.load_record(module.into_record());
                self = self.refresh();
            }
            RecordState::Loaded(base) => {
                self.module = self.module.load_record(base);
                self = self.refresh();
            }
        }
        self.bind()
    }
}

impl<B, M, R> AutodiffModule<B> for Reparameterized<M, R, B>
where
    B: AutodiffBackend, M: AutodiffModule<B>, R: Reparameterization<B> + AutodiffModule<B>,
    R::InnerModule: Reparameterization<B::InnerBackend>,
{
    type InnerModule = Reparameterized<M::InnerModule, R::InnerModule, B::InnerBackend>;
    fn valid(&self) -> Self::InnerModule {
        Reparameterized { module: self.module.valid(), owner: self.owner, backend: PhantomData,
            attachments: self.attachments.iter().map(|attachment| Attachment {
                path: attachment.path.clone(), state: attachment.state.valid(),
            }).collect() }.bind()
    }
    fn from_inner(module: Self::InnerModule) -> Self {
        Reparameterized { module: M::from_inner(module.module), owner: module.owner, backend: PhantomData,
            attachments: module.attachments.into_iter().map(|attachment| Attachment {
                path: attachment.path, state: R::from_inner(attachment.state),
            }).collect() }.bind()
    }
}

impl<B, M, R> HasAutodiffModule<B> for Reparameterized<M, R, B::InnerBackend>
where
    B: AutodiffBackend, M: Module<B::InnerBackend> + HasAutodiffModule<B>,
    R: Reparameterization<B::InnerBackend> + HasAutodiffModule<B>,
    R::TrainModule: Reparameterization<B>,
{
    type TrainModule = Reparameterized<M::TrainModule, R::TrainModule, B>;
}

impl<M: ModuleDisplay, R, B: Backend> ModuleDisplayDefault for Reparameterized<M, R, B> {
    fn content(&self, content: Content) -> Option<Content> { content.add_single(&self.module).optional() }
}
impl<M: ModuleDisplay, R, B: Backend> ModuleDisplay for Reparameterized<M, R, B> {}

enum RecordState<M: Module<B>, R: Reparameterization<B>, B: Backend> {
    Live(M, Vec<Attachment<R>>, usize),
    Loaded(M::Record),
}

/// A structural model record including each attached module's own record.
pub struct ReparameterizedRecord<M: Module<B>, R: Reparameterization<B>, B: Backend> {
    state: RecordState<M, R, B>,
}

impl<B: Backend, M: Module<B>, R: Reparameterization<B>> Record<B> for ReparameterizedRecord<M, R, B> {
    type Item<S: PrecisionSettings> = <M::Record as Record<B>>::Item<S>;
    fn map_parameters<V: ModuleMapper<B>>(self, mapper: &mut V) -> Self {
        Self { state: match self.state {
            RecordState::Live(module, mut attachments, owner) => {
                let module = module.map(&mut Map { attachments: &mut attachments, mapper, path: Vec::new(), owner });
                RecordState::Live(module, attachments, owner)
            }
            RecordState::Loaded(base) => RecordState::Loaded(base.map_parameters(mapper)),
        } }
    }

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        match self.state {
            RecordState::Live(module, _, _) => module.into_record().into_item::<S>(),
            RecordState::Loaded(base) => base.into_item::<S>(),
        }
    }
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Self { state: RecordState::Loaded(M::Record::from_item::<S>(item, device)) }
    }
}

impl<B: Backend, M: Module<B>, R: Reparameterization<B>> RuntimeRecord<B> for ReparameterizedRecord<M, R, B>
where M::Record: RuntimeRecord<B> {
    fn into_runtime(self, precision: RuntimePrecision) -> crate::record::RuntimeRecordValue {
        match self.state {
            RecordState::Live(module, _, _) => module.into_record().into_runtime(precision),
            RecordState::Loaded(base) => base.into_runtime(precision),
        }
    }
    fn from_runtime(value: crate::record::RuntimeRecordValue, device: &B::Device) -> Self {
        Self { state: RecordState::Loaded(M::Record::from_runtime(value, device)) }
    }
}
