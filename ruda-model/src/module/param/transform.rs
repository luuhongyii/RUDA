use super::{Parameter, ParamId};
use core::any::Any;

#[cfg(target_has_atomic = "ptr")]
pub(crate) type SharedTransform<T> = alloc::sync::Arc<dyn ParamTransform<T>>;
#[cfg(not(target_has_atomic = "ptr"))]
pub(crate) type SharedTransform<T> = portable_atomic_util::Arc<alloc::boxed::Box<dyn ParamTransform<T>>>;

pub(crate) fn share<T: Parameter, R: ParamTransform<T> + 'static>(value: R) -> SharedTransform<T> {
    #[cfg(target_has_atomic = "ptr")]
    { alloc::sync::Arc::new(value) }
    #[cfg(not(target_has_atomic = "ptr"))]
    { portable_atomic_util::Arc::new(alloc::boxed::Box::new(value)) }
}

pub(crate) trait ParamTransform<T: Parameter>: core::fmt::Debug + Send + Sync {
    fn name(&self) -> &'static str;
    fn apply(&self, base: T) -> T;
    fn as_any(&self) -> &dyn Any;
    fn owner(&self) -> usize;
    fn to_device(&self, device: &T::Device) -> SharedTransform<T>;
    fn fork(&self, device: &T::Device) -> SharedTransform<T>;
    fn collect_devices(&self, devices: alloc::vec::Vec<T::Device>) -> alloc::vec::Vec<T::Device>;
    fn load(&self, state: &dyn Any) -> SharedTransform<T>;
    fn record(&self, precision: crate::record::RuntimePrecision) -> TransformRecord;
    fn load_record(&self, record: &TransformRecord, device: &T::Device) -> SharedTransform<T>;
    fn visit_dyn(&self, visitor: &mut dyn super::transform_traversal::DynVisitor);
    fn map_dyn(&self, mapper: &mut dyn super::transform_traversal::DynMapper) -> SharedTransform<T>;
    fn visit_float_primitives(&self, visitor: &mut dyn FnMut(ParamId, &dyn Any));
}

/// Serialized state attached to a structural parameter in a model record.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct TransformRecord {
    pub name: alloc::string::String,
    pub data: alloc::vec::Vec<u8>,
    #[serde(skip)]
    pub precision: Option<core::any::TypeId>,
}
