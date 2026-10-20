use super::ParamId;
use super::sync_once_cell::SyncOnceCell;
use alloc::format;

#[cfg(not(target_has_atomic = "ptr"))]
use alloc::boxed::Box;
use ruda_core::stub::{Mutex, RwLock};
use ruda_tensor::api::Shape;
use core::ops::Deref;

#[cfg(target_has_atomic = "ptr")]
use alloc::sync::Arc;

#[cfg(not(target_has_atomic = "ptr"))]
use portable_atomic_util::Arc;

#[cfg(target_has_atomic = "ptr")]
type Mapper<T> = Arc<dyn Fn(T) -> T + Send + Sync>;

#[cfg(not(target_has_atomic = "ptr"))]
type Mapper<T> = Arc<Box<dyn Fn(T) -> T + Send + Sync>>;

#[cfg(target_has_atomic = "ptr")]
fn new_mapper<T, F: Fn(T) -> T + Send + Sync + 'static>(func: F) -> Mapper<T> {
    Arc::new(func)
}

#[cfg(not(target_has_atomic = "ptr"))]
fn new_mapper<T, F: Fn(T) -> T + Send + Sync + 'static>(func: F) -> Mapper<T> {
    Arc::new(Box::new(func))
}

/// Type alias for the init function stored in `Uninitialized`.
/// On targets without atomics, `portable_atomic_util::Arc` needs `Box` indirection
/// for unsized types, mirroring the `Mapper` pattern above.
#[cfg(target_has_atomic = "ptr")]
type InitFn<P> = Arc<dyn Fn(&<P as Parameter>::Device, bool) -> P + Send + Sync>;

#[cfg(not(target_has_atomic = "ptr"))]
type InitFn<P> = Arc<Box<dyn Fn(&<P as Parameter>::Device, bool) -> P + Send + Sync>>;

#[cfg(target_has_atomic = "ptr")]
fn new_init_fn<P: Parameter, F: Fn(&P::Device, bool) -> P + Send + Sync + 'static>(
    func: F,
) -> InitFn<P> {
    Arc::new(func)
}

#[cfg(not(target_has_atomic = "ptr"))]
fn new_init_fn<P: Parameter, F: Fn(&P::Device, bool) -> P + Send + Sync + 'static>(
    func: F,
) -> InitFn<P> {
    Arc::new(Box::new(func))
}

struct SharedInitializer<P: Parameter> {
    init: InitFn<P>,
    value: SyncOnceCell<Mutex<(P, bool)>>,
}

impl<P: Parameter> SharedInitializer<P> {
    fn new(init: InitFn<P>) -> Arc<Self> {
        Arc::new(Self {
            init,
            value: SyncOnceCell::new(),
        })
    }

    fn initialize(&self, device: &P::Device, require_grad: bool) -> P {
        let cached = self
            .value
            .get_or_init(|| Mutex::new(((self.init)(device, require_grad), require_grad)))
            .lock()
            .unwrap();
        let (value, initialized_require_grad) = (cached.0.clone(), cached.1);
        core::mem::drop(cached);
        if require_grad == initialized_require_grad {
            value
        } else {
            value.set_require_grad(require_grad)
        }
    }

    fn is_initialized(&self) -> bool {
        self.value.get().is_some()
    }
}

/// Parameters are the fundamental building blocks of [modules](crate::module::Module) where they
/// serve as containers for [tensors](crate::tensor::Tensor) that can be updated during
/// training, and loaded during inference. If you don't want to save the tensors
/// and/or don't want to update it during training, you don't need this type to wrap your tensor.
///
/// # Core Lazy Initialization Architecture
///
/// `Param<T>` has a dual-state design using `SyncOnceCell<T>`:
///
/// ## State Management
///
/// **Two possible states:**
///
/// 1. **Initialized**: `state: SyncOnceCell<T>` contains value, `initialization: None`
/// 2. **Uninitialized (Lazy)**: `state` is empty, `initialization: Some(RwLock<Option<Uninitialized<T>>>)`
pub struct Param<T: Parameter> {
    /// The unique ID of this parameter. This is used by eg. optimizers to associate a gradient with a specific parameter.
    pub id: ParamId,
    /// The SyncOnceCell holding the initialized parameter value.
    /// Empty for uninitialized parameters, populated after first access or explicit initialization.
    pub(crate) state: SyncOnceCell<T>,
    /// The deferred initialization state for lazy parameters.
    ///
    /// **State Transitions:**
    /// - Initialized params: `None`
    /// - Uninitialized params: `Some(RwLock<Some(Uninitialized<T>)>)`
    /// - After lazy init triggers: `Some(RwLock<None>)` (inner Option is taken)
    pub(crate) initialization: Option<RwLock<Option<Uninitialized<T>>>>,
    pub(crate) param_mapper: ParamMapper<T>,
    // For stateful `module.valid()` <> `module.train()`
    pub(crate) require_grad: bool,
    pub(crate) transform: Option<super::transform::SharedTransform<T>>,
    pub(crate) transform_record: Option<super::transform::TransformRecord>,
}

#[derive(Clone)]
/// Applies transformations when loading and saving parameters.
///
/// # Mapper System
///
/// `ParamMapper<T>` allows applying transformations during serialization and deserialization:
/// - `load: Option<Mapper<T>>` - transformation during deserialization (applied in `transform_for_load()`)
/// - `save: Option<Mapper<T>>` - transformation during serialization (applied in `transform_for_save()`)
///
/// These are commonly used for:
/// - Quantization/dequantization
/// - Precision conversion (e.g., FP32 ↔ FP16)
/// - Custom parameter transformations
pub struct ParamMapper<T: Parameter> {
    load: Option<Mapper<T>>,
    save: Option<Mapper<T>>,
    mapped_require_grad: Option<bool>,
}

impl<T: Parameter> core::fmt::Debug for ParamMapper<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!(
            "ParamMapper {{ load: {}, save: {} }}",
            self.load.is_some(),
            self.save.is_some()
        ))
    }
}

impl<T: Parameter> ParamMapper<T> {
    pub(crate) fn convert<U, IntoOld, IntoNew>(self, into_old: IntoOld, into_new: IntoNew) -> ParamMapper<U>
    where
        T: 'static, U: Parameter + 'static,
        IntoOld: Fn(U) -> T + Clone + Send + Sync + 'static,
        IntoNew: Fn(T) -> U + Clone + Send + Sync + 'static,
    {
        let load_old = into_old.clone();
        let load_new = into_new.clone();
        ParamMapper {
            load: self.load.map(|mapper| new_mapper(move |value| load_new(mapper(load_old(value))))),
            save: self.save.map(|mapper| new_mapper(move |value| into_new(mapper(into_old(value))))),
            mapped_require_grad: self.mapped_require_grad,
        }
    }

    /// Applies the transformation when loading the given parameter.
    pub fn on_load(&self, param: T) -> T {
        match &self.load {
            Some(mapper) => mapper(param),
            None => param,
        }
    }
    /// Applies the transformation when saving the given parameter.
    pub fn on_save(&self, param: T) -> T {
        match &self.save {
            Some(mapper) => mapper(param),
            None => param,
        }
    }
}

impl<T: Parameter> Default for ParamMapper<T> {
    fn default() -> Self {
        Self {
            load: None,
            save: None,
            mapped_require_grad: None,
        }
    }
}

impl<T: Parameter> core::fmt::Display for Param<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(format!("Param: {}", self.id).as_str())
    }
}

impl<T: Parameter> core::fmt::Debug for Param<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(format!("Param: {} - {:?}", self.id, self.param_mapper).as_str())
    }
}

/// Trait that defines what is necessary for a type to be a parameter.
pub trait Parameter: Clone + core::fmt::Debug + Send {
    /// The device type to be used.
    type Device: Clone;

    /// Fetch the device.
    fn device(&self) -> Self::Device;

    /// Fetch the gradient requirement.
    fn is_require_grad(&self) -> bool;

    /// Set the gradient requirement.
    fn set_require_grad(self, require_grad: bool) -> Self;

    #[doc(hidden)]
    fn into_materialized(self, require_grad: bool) -> Self { self.set_require_grad(require_grad) }

    #[doc(hidden)]
    fn mapped_require_grad(&self, _previous: bool) -> bool {
        self.is_require_grad()
    }

    #[doc(hidden)]
    fn autodiff_context(&self) -> Option<ruda_tensor::AutodiffContext> { None }

    #[doc(hidden)]
    fn creation_context(_device: &Self::Device) -> Option<ruda_tensor::AutodiffContext> { None }

    #[doc(hidden)]
    fn in_creation_context<R>(_device: &Self::Device, _context: Option<ruda_tensor::AutodiffContext>, func: impl FnOnce() -> R) -> R { func() }
}

/// The deferred initialization state for lazy parameters.
#[allow(clippy::type_complexity)]
pub(crate) struct Uninitialized<P: Parameter> {
    /// The initialization function. Called with `(device, is_require_grad) -> Parameter`.
    /// Clones share the first initialization result without triggering evaluation.
    /// Per-parameter transformations and gradient settings remain independent.
    init: Arc<SharedInitializer<P>>,
    /// The target device on which the parameter should be initialized.
    /// Used by `lazy_device()` to provide device information without triggering initialization.
    pub(crate) device: P::Device,
    /// The gradient requirement for the parameter.
    /// Used by `lazy_is_require_grad()` to provide gradient settings without triggering initialization.
    pub(crate) is_require_grad: bool,
    /// The shape of the tensor parameter.
    /// Used by `lazy_shape()` to provide shape information without triggering initialization.
    pub(crate) shape: Shape,
    context: Option<ruda_tensor::AutodiffContext>,
}

impl<P: Parameter> Clone for Uninitialized<P> {
    fn clone(&self) -> Self {
        Self {
            init: self.init.clone(),
            device: self.device.clone(),
            is_require_grad: self.is_require_grad,
            shape: self.shape.clone(),
            context: self.context,
        }
    }
}

impl<P: Parameter> Uninitialized<P> {
    /// Runs the initialization function.
    ///
    /// This is called by [Param::val] when accessing an uninitialized parameter for the first time.
    /// The function is given the stored device and gradient requirement, and returns the initialized parameter.
    ///
    /// The initializer is evaluated once across all clones; each parameter caches its own value.
    fn initialize(&self) -> P {
        self.init.initialize(&self.device, self.is_require_grad)
    }
}

impl<T: Parameter> Param<T> {
    /// Create a new parameter that is already initialized.
    pub fn initialized(id: ParamId, value: T) -> Self {
        let require_grad = value.is_require_grad();
        Self {
            id,
            state: SyncOnceCell::initialized(value),
            initialization: None,
            param_mapper: Default::default(),
            require_grad,
            transform: None,
            transform_record: None,
        }
    }

    /// Create a new parameter that is not already initialized.
    pub fn uninitialized<F>(
        id: ParamId,
        init: F,
        device: T::Device,
        is_require_grad: bool,
        shape: Shape,
    ) -> Self
    where
        F: Fn(&T::Device, bool) -> T + Send + Sync + 'static,
    {
        let context = T::creation_context(&device);
        Self {
            id,
            state: SyncOnceCell::new(),
            initialization: Some(RwLock::new(Some(Uninitialized {
                init: SharedInitializer::new(new_init_fn(move |device, require_grad| {
                    T::in_creation_context(device, context, || init(device, require_grad))
                })),
                device,
                is_require_grad,
                shape,
                context,
            }))),
            param_mapper: Default::default(),
            require_grad: is_require_grad,
            transform: None,
            transform_record: None,
        }
    }

    /// Gets the parameter value, initializing it lazily if needed.
    ///
    /// For initialized parameters, this returns a clone of the cached value.
    /// For uninitialized parameters, this triggers initialization:
    pub fn val(&self) -> T {
        let base = self.base();
        match &self.transform {
            Some(transform) => transform.apply(base),
            None => base,
        }
    }

    /// Return the stored base without materializing an attached transformation.
    pub fn base(&self) -> T {
        self.state
            .get_or_init(|| {
                let mut result = self
                    .initialization
                    .as_ref()
                    .expect("Should have an initialization when no state provided.")
                    .write()
                    .unwrap();
                let state = result.take().expect("Should exist when not initialized");
                state.initialize()
            })
            .clone()
    }

    /// Borrow the concrete state attached by a reparameterized module.
    pub fn reparameterization<R: 'static>(&self) -> Option<&R> {
        self.transform.as_ref()?.as_any().downcast_ref()
    }

    #[doc(hidden)]
    pub fn visit_reparameterization_primitives(&self, visitor: &mut dyn FnMut(ParamId, &dyn core::any::Any)) {
        if let Some(transform) = &self.transform { transform.visit_float_primitives(visitor); }
    }

    pub(crate) fn transform_owner(&self) -> Option<usize> {
        self.transform.as_ref().map(|transform| transform.owner())
    }

    pub(crate) fn without_transform(mut self) -> Self {
        self.transform = None;
        self
    }

    /// Replace the structural base with its current effective value.
    pub fn materialize(self) -> Self {
        if self.transform.is_none() { return self; }
        let value = self.val().into_materialized(self.base().is_require_grad());
        let mut result = Self::from_mapped_value(self.id, value, self.param_mapper);
        result.require_grad = self.require_grad;
        result
    }

    /// Check if the parameter has been initialized.
    ///
    /// Returns `true` if the parameter's value has been computed and cached,
    /// `false` if it's still lazy and will be initialized on first access.
    pub fn is_initialized(&self) -> bool {
        self.state.get().is_some()
            || self.initialization.as_ref().is_some_and(|init| {
                init.read()
                    .unwrap()
                    .as_ref()
                    .is_some_and(|value| value.init.is_initialized())
            })
    }

    /// Read the initialized or declared lazy gradient requirement without running the initializer.
    /// A backend/device without autodiff can still produce an untracked tensor when initialized.
    pub fn planned_is_require_grad(&self) -> bool {
        self.require_grad
    }

    /// Gets the parameter's value while consuming the parameter.
    pub fn into_value(self) -> T {
        self.consume().1
    }

    /// Gets the parameter id and value while consuming the parameter.
    pub fn consume(self) -> (ParamId, T, ParamMapper<T>) {
        let tensor = self.base();
        let mut mapper = self.param_mapper;
        mapper.mapped_require_grad = Some(self.require_grad);

        core::mem::drop(self.state);

        (self.id, tensor, mapper)
    }

    /// Execute the given function on the inner value.
    pub fn map<F: FnOnce(T) -> T>(mut self, func: F) -> Self {
        let previous = self.require_grad;
        let transform = self.transform.take();
        let transform_record = self.transform_record.take();
        let (id, tensor, param_mapper) = self.consume();
        let tensor = func(tensor);
        let require_grad = tensor.mapped_require_grad(previous);

        Self {
            id,
            state: SyncOnceCell::initialized(tensor),
            initialization: None,
            param_mapper,
            require_grad,
            transform,
            transform_record,
        }
    }

    /// Create an initialized parameter with the given id, value, and param mapper.
    ///
    /// This is a helper method for creating parameters while preserving the param mapper,
    /// typically used in ModuleMapper implementations.
    pub fn from_mapped_value(id: ParamId, value: T, mut param_mapper: ParamMapper<T>) -> Self {
        let require_grad = param_mapper.mapped_require_grad.take().unwrap_or_else(|| value.is_require_grad());
        Self {
            id,
            state: SyncOnceCell::initialized(value),
            initialization: None,
            param_mapper,
            require_grad,
            transform: None,
            transform_record: None,
        }
    }

    /// Runs a transformation on the parameter when loading.
    pub fn load_mapper<F: Fn(T) -> T + Send + Sync + 'static>(mut self, func: F) -> Self {
        self.param_mapper.load = Some(new_mapper(func));

        self
    }

    /// Runs a transformation on the parameter when saving.
    pub fn save_mapper<F: Fn(T) -> T + Send + Sync + 'static>(mut self, func: F) -> Self {
        self.param_mapper.save = Some(new_mapper(func));

        self
    }

    /// Execute the given function on the inner value.
    pub fn init_mapper<F: Fn(T) -> T + Send + Sync + 'static>(self, func: F) -> Self
    where
        T: 'static,
    {
        let initialization = match &self.initialization {
            Some(init) => init,
            None => return self.map(func),
        };

        let mut init = initialization.write().unwrap();

        match init.as_mut() {
            Some(value) => {
                if value.init.is_initialized() {
                    core::mem::drop(init);
                    return self.map(func);
                }
                let prev = value.init.clone();

                value.init = SharedInitializer::new(new_init_fn(move |a, b| {
                    let tensor = prev.initialize(a, b);
                    func(tensor)
                }));
                core::mem::drop(init);
                self
            }
            None => {
                core::mem::drop(init);
                self.map(func)
            }
        }
    }

    /// The device on which the parameter is or will be initialized, **without triggering initialization**.
    ///
    /// This is critical for the load optimization: when loading tensors into an uninitialized parameter,
    /// we need to know the target device to move the loaded tensor appropriately, but we don't want to
    /// trigger the initialization function (which would allocate an unnecessary tensor).
    ///
    /// Use this instead of [crate::tensor::Tensor::device] when you need the device but want to
    /// preserve lazy initialization.
    pub fn lazy_device(&self) -> T::Device {
        let initialization = match &self.initialization {
            Some(init) => init,
            None => return self.device(),
        };

        let init = initialization.read().unwrap();

        match init.as_ref() {
            Some(value) => value.device.clone(),
            None => self.device(),
        }
    }

    /// The gradient requirement on which the parameter is or will be initialized, **without triggering initialization**.
    ///
    /// Similar to [lazy_device](Self::lazy_device), this is critical for the load optimization.
    /// When loading tensors into an uninitialized parameter, we need to apply the correct gradient
    /// setting to the loaded tensor without triggering the initialization function.
    ///
    /// # Notes
    ///
    /// This is a crate-private function, since users are not expected to use `is_require_grad` of an
    /// uninitialized module to then override its value. All low-level functions should be provided
    /// by `ruda` and should handle those details.
    pub(crate) fn lazy_is_require_grad(&self) -> bool {
        let initialization = match &self.initialization {
            Some(init) => init,
            None => return self.is_require_grad(),
        };

        let init = initialization.read().unwrap();

        match init.as_ref() {
            Some(value) => value.is_require_grad,
            None => self.is_require_grad(),
        }
    }

    pub(crate) fn lazy_autodiff_context(&self) -> Option<ruda_tensor::AutodiffContext> {
        if let Some(value) = self.state.get() { return value.autodiff_context(); }
        let context = self.initialization.as_ref().and_then(|init| {
            init.read().unwrap().as_ref().map(|init| init.context)
        });
        match context {
            Some(context) => context,
            None => self.deref().autodiff_context(),
        }
    }

    /// Override the gradient requirement for the current parameter.
    pub fn set_require_grad(mut self, require_grad: bool) -> Self {
        self.require_grad = require_grad;
        let initialization = match &self.initialization {
            Some(init) => init,
            None => {
                let mut param = self.map(|tensor| tensor.set_require_grad(require_grad));
                param.require_grad = require_grad;
                return param;
            }
        };

        let mut init = initialization.write().unwrap();
        let mut is_lazy = false;

        if let Some(value) = init.as_mut() {
            is_lazy = true;
            value.is_require_grad = require_grad;
        };

        core::mem::drop(init);

        if is_lazy {
            return self;
        }

        let mut param = self.map(|tensor| tensor.set_require_grad(require_grad));
        param.require_grad = require_grad;
        param
    }
}

impl<T: Parameter> Clone for Param<T> {
    fn clone(&self) -> Self {
        // If uninitialized, clone the lazy state without triggering initialization.
        // This avoids allocating tensor memory for params that may never be used
        // (e.g., when cloning a module just to load weights into it).
        // The local value and metadata are independent; the initializer result is shared.
        if let Some(init_lock) = &self.initialization {
            let init_guard = init_lock.read().unwrap();
            if let Some(uninit) = init_guard.as_ref() {
                return Self {
                    id: self.id,
                    state: SyncOnceCell::new(),
                    initialization: Some(RwLock::new(Some(uninit.clone()))),
                    param_mapper: self.param_mapper.clone(),
                    require_grad: self.require_grad,
                    transform: self.transform.clone(),
                    transform_record: self.transform_record.clone(),
                };
            }
        }

        // Already initialized (or init was already consumed): clone the value.
        let mut param = Param::initialized(self.id, self.base());
        param.param_mapper = self.param_mapper.clone();
        param.require_grad = self.require_grad;
        param.transform = self.transform.clone();
        param.transform_record = self.transform_record.clone();
        param
    }
}

impl<T: Parameter> Deref for Param<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.state.get_or_init(|| {
            let mut result = self
                .initialization
                .as_ref()
                .expect("Should have an initialization when no state provided.")
                .write()
                .unwrap();

            let state = result.take().expect("Should exist when not initialized");
            state.initialize()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruda_tensor::api::{Tensor, backend::Backend};

    // Param<T> should be Sync so that models can be shared across threads
    // (e.g. parallel inference with rayon).
    fn _assert_sync<T: Sync>() {}

    #[test]
    fn param_is_sync() {
        fn check<B: Backend>() {
            _assert_sync::<Param<Tensor<B, 2>>>();
        }
        check::<crate::TestBackend>();
    }

    /// Concurrent lazy initialization must not panic.
    ///
    /// Multiple threads call `val()` on an uninitialized `Param` simultaneously.
    /// `SyncOnceCell::get_or_init` guarantees only one thread runs the initializer;
    /// the others block and receive the same value.
    #[cfg(feature = "std")]
    #[test]
    fn param_concurrent_lazy_init() {
        use alloc::vec::Vec;

        type B = crate::TestBackend;
        let device = Default::default();

        let param: Param<Tensor<B, 2>> = Param::uninitialized(
            ParamId::new(),
            |device, _require_grad| Tensor::zeros([2, 3], device),
            device,
            false,
            [2, 3].into(),
        );

        // Share across threads via &param (requires Sync).
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..4).map(|_| s.spawn(|| param.val())).collect();

            let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

            // All threads must get the same value.
            let expected = results[0].to_data();
            for result in &results[1..] {
                assert_eq!(result.to_data(), expected);
            }
        });
    }

    #[cfg(feature = "std")]
    #[test]
    fn param_clones_initialize_once_across_threads() {
        use alloc::vec::Vec;
        use core::sync::atomic::{AtomicUsize, Ordering};

        type B = crate::TestBackend;
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let param: Param<Tensor<B, 1>> = Param::uninitialized(
            ParamId::new(),
            move |device, _| {
                let value = counter.fetch_add(1, Ordering::SeqCst) + 1;
                Tensor::full([4], value as f32, device)
            },
            Default::default(),
            false,
            [4].into(),
        );
        let clones: Vec<_> = (0..8).map(|_| param.clone()).collect();
        assert!(!param.is_initialized());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        std::thread::scope(|scope| {
            let handles: Vec<_> = clones
                .iter()
                .map(|clone| scope.spawn(move || clone.val().to_data()))
                .collect();
            for handle in handles {
                assert_eq!(handle.join().unwrap().to_vec::<f32>().unwrap(), [1.; 4]);
            }
        });
        assert!(param.is_initialized());
        assert_eq!(param.val().to_data().to_vec::<f32>().unwrap(), [1.; 4]);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn param_lazy_mapping_shares_base_but_not_transformation() {
        use core::sync::atomic::{AtomicUsize, Ordering};

        type B = crate::TestBackend;
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let param: Param<Tensor<B, 1>> = Param::uninitialized(
            ParamId::new(),
            move |device, _| {
                let value = counter.fetch_add(1, Ordering::SeqCst) + 1;
                Tensor::full([4], value as f32, device)
            },
            Default::default(),
            false,
            [4].into(),
        );
        let mapped = param.clone().init_mapper(|tensor| tensor * 2.);
        let mapped_clone = mapped.clone();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(!param.is_initialized());
        assert!(!mapped.is_initialized());
        assert_eq!(
            mapped_clone.val().to_data().to_vec::<f32>().unwrap(),
            [2.; 4]
        );
        assert_eq!(mapped.val().to_data().to_vec::<f32>().unwrap(), [2.; 4]);
        assert!(param.is_initialized());
        assert_eq!(param.val().to_data().to_vec::<f32>().unwrap(), [1.; 4]);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let mapped = param.clone().init_mapper(|tensor| tensor + 3.);
        assert!(mapped.is_initialized());
        assert_eq!(mapped.val().to_data().to_vec::<f32>().unwrap(), [4.; 4]);
        assert_eq!(param.val().to_data().to_vec::<f32>().unwrap(), [1.; 4]);
    }

    #[test]
    fn param_lazy_clone_gradient_settings_are_independent() {
        type B = crate::TestAutodiffBackend;
        for clone_first in [false, true] {
            let param: Param<Tensor<B, 1>> = Param::uninitialized(
                ParamId::new(),
                |device, require_grad| {
                    Tensor::random([4], Default::default(), device).set_require_grad(require_grad)
                },
                Default::default(),
                true,
                [4].into(),
            );
            let detached = param.clone().set_require_grad(false);
            assert!(!param.is_initialized());
            assert!(!detached.is_initialized());
            assert!(param.planned_is_require_grad());
            assert!(!detached.planned_is_require_grad());
            if clone_first {
                let _ = detached.val();
            } else {
                let _ = param.val();
            }
            assert!(param.val().is_require_grad());
            assert!(!detached.val().is_require_grad());
            assert_eq!(param.val().to_data(), detached.val().to_data());
        }
    }

    #[test]
    fn param_loading_a_lazy_clone_does_not_initialize_or_replace_its_peer() {
        use crate::module::Module;
        use core::sync::atomic::{AtomicUsize, Ordering};

        type B = crate::TestBackend;
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let param: Param<Tensor<B, 1>> = Param::uninitialized(
            ParamId::new(),
            move |device, _| {
                counter.fetch_add(1, Ordering::SeqCst);
                Tensor::ones([4], device)
            },
            Default::default(),
            false,
            [4].into(),
        );
        let loaded = param.clone().load_record(Param::initialized(
            param.id,
            Tensor::full([4], 7., &Default::default()),
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(!param.is_initialized());
        assert_eq!(loaded.val().to_data().to_vec::<f32>().unwrap(), [7.; 4]);
        assert_eq!(param.val().to_data().to_vec::<f32>().unwrap(), [1.; 4]);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
