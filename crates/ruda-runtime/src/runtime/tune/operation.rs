use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt::{Debug, Display, Write};
use core::hash::Hash;
use spin::Once;

use alloc::format;

use super::{
    AutotuneError, input_generator::InputGenerator, key_generator::KeyGenerator,
    tune_inputs::TuneInputs,
};
use super::{Bounds, BoundsGenerator, Eviction, Evictor, Tunable, TunePlan};

/// A type-erased delegate for a tunable function.
///
/// The lifetime `'inp` is the lifetime of the input data, the function must be defined such that
/// it can be called for any lifetime `inp` and produce a `Result<Out, AutotuneError>`.
type TuneDelegate<I, Out> =
    dyn for<'inp> Fn(<I as TuneInputs>::At<'inp>) -> Result<Out, AutotuneError> + Send + Sync;

/// A named, type-erased tunable function stored in a [`TunableSet`]. Constructed via
/// [`Tunable::new`](super::Tunable::new); callers don't name this type directly.
#[derive(new)]
pub struct TuneFn<I: TuneInputs, Out> {
    pub(crate) name: String,
    func: Box<TuneDelegate<I, Out>>,
}

impl<I: TuneInputs, Out: 'static> TuneFn<I, Out> {
    /// Run the wrapped function on the given inputs.
    pub fn execute<'a>(&self, inputs: <I as TuneInputs>::At<'a>) -> Result<Out, AutotuneError> {
        (self.func)(inputs)
    }
}

/// A set of candidate tunable functions for autotune, sharing a key generator and an
/// input generator. See [`TuneInputs`] for the `F` parameter.
pub struct TunableSet<K: AutotuneKey, F: TuneInputs, Output: 'static, Id = ()> {
    tunables: Vec<Tunable<K, F, Output, Id>>,
    key_gen: Arc<dyn KeyGenerator<K, F> + Send + Sync>,
    input_gen: Arc<dyn InputGenerator<K, F> + Send + Sync>,
    bounds_gen: Option<Arc<dyn BoundsGenerator<K, F> + Send + Sync>>,
    eviction: Option<Arc<dyn Eviction<K, F> + Send + Sync>>,
    short_circuit: bool,
    stack_reference: Option<usize>,
    stack_revision: String,
    stack_workload: Option<Arc<dyn for<'a> Fn(&F::At<'a>) -> String + Send + Sync>>,
    checksum: Once<String>,
    stack_manifest: Once<String>,
    #[cfg(std_io)]
    stack_plan_cache: Option<spin::Mutex<alloc::collections::VecDeque<(K, String, Arc<StaticStackPlan>)>>>,
}

#[cfg(std_io)]
pub(crate) struct StaticStackPlan {
    pub(crate) indices: Vec<usize>,
    pub(crate) candidates: Vec<super::stack::Candidate>,
    pub(crate) key: String,
}

impl<K: AutotuneKey, F: TuneInputs, Output: 'static, Id> TunableSet<K, F, Output, Id> {
    /// The number of tunables in the set.
    pub fn len(&self) -> usize {
        self.tunables.len()
    }

    /// Whether this set contains no tunables.
    pub fn is_empty(&self) -> bool {
        self.tunables.is_empty()
    }

    /// Create a tunable set from a key generator and an input generator.
    pub fn new(key_gen: impl KeyGenerator<K, F>, input_gen: impl InputGenerator<K, F>) -> Self {
        Self {
            tunables: Default::default(),
            input_gen: Arc::new(input_gen),
            key_gen: Arc::new(key_gen),
            bounds_gen: None,
            eviction: None,
            short_circuit: true,
            stack_reference: None,
            stack_revision: String::new(),
            stack_workload: None,
            checksum: Once::new(),
            stack_manifest: Once::new(),
            #[cfg(std_io)]
            stack_plan_cache: None,
        }
    }

    /// Register an exact workload signature and a known-safe reference for full-stack tuning.
    /// The input generator MUST isolate all mutable state for each benchmark invocation.
    /// Include dtype, exact shape/strides, precision, optional inputs and semantic parameters.
    pub fn with_stack_tuning(
        mut self, reference: usize, revision: &str,
        workload: impl for<'a> Fn(&F::At<'a>) -> String + Send + Sync + 'static,
    ) -> Self {
        self.stack_reference = Some(reference);
        self.stack_revision = revision.into();
        self.stack_workload = Some(Arc::new(workload));
        self.stack_manifest = Once::new();
        #[cfg(std_io)]
        if let Some(cache) = &self.stack_plan_cache { cache.lock().clear(); }
        self
    }
    pub fn stack_reference(&self) -> Option<usize> { self.stack_reference }
    pub fn stack_workload<'a>(&self, inputs: &F::At<'a>) -> Option<String> {
        self.stack_workload.as_ref().map(|f| f(inputs))
    }

    /// Cache plan metadata when priorities depend only on the generated key and immutable set.
    /// This does not cache a winner or bypass normal selection, validation or expiration.
    pub fn with_static_stack_plan(mut self) -> Self {
        #[cfg(std_io)]
        { self.stack_plan_cache = Some(spin::Mutex::new(Default::default())); }
        self
    }

    #[cfg(std_io)]
    pub(crate) fn static_stack_plan_enabled(&self) -> bool { self.stack_plan_cache.is_some() }

    #[cfg(std_io)]
    pub(crate) fn cached_stack_plan(&self, key: &K, identity: &str) -> Option<Arc<StaticStackPlan>> {
        self.stack_plan_cache.as_ref()?.lock().iter()
            .find(|(stored_key, stored_identity, _)| stored_key == key && stored_identity == identity)
            .map(|(_, _, plan)| plan.clone())
    }

    #[cfg(std_io)]
    pub(crate) fn cache_stack_plan(&self, key: K, identity: String, plan: Arc<StaticStackPlan>) {
        if let Some(cache) = &self.stack_plan_cache {
            let mut cache = cache.lock();
            if cache.iter().any(|(stored_key, stored_identity, _)| stored_key == &key && stored_identity == &identity) { return; }
            if cache.len() == 32 { cache.pop_front(); }
            cache.push_back((key, identity, plan));
        }
    }

    /// Shorthand for [`new`](Self::new) with a [`CloneInputGenerator`]: benchmarks run
    /// on clones of the real call inputs.
    pub fn new_cloning_inputs(key_gen: impl KeyGenerator<K, F>) -> Self {
        Self::new(key_gen, super::CloneInputGenerator)
    }

    pub fn with_bounds(mut self, bounds: Arc<dyn BoundsGenerator<K, F> + Send + Sync>) -> Self {
        self.bounds_gen = Some(bounds);
        self
    }

    pub fn with_eviction(mut self, eviction: Arc<dyn Eviction<K, F> + Send + Sync>) -> Self {
        self.eviction = Some(eviction);
        self
    }

    pub fn with_short_circuit(mut self, enabled: bool) -> Self {
        self.short_circuit = enabled;
        self
    }

    pub fn is_short_circuit_enabled(&self) -> bool { self.short_circuit }

    pub fn bounds(&self, key: &K, inputs: &F::At<'_>) -> Option<Bounds> {
        self.bounds_gen.as_ref().map(|bounds| bounds.generate(key, inputs))
    }

    pub(crate) fn evictor<'i>(&self, key: &K, inputs: &F::At<'i>) -> Option<Box<Evictor<'i>>> {
        let eviction = self.eviction.clone()?;
        let key = key.clone();
        let inputs = inputs.clone();
        Some(Box::new(move || eviction.evict(&key, &inputs)))
    }

    /// Register a tunable with this tunable set.
    pub fn with(mut self, tunable: Tunable<K, F, Output, Id>) -> Self {
        self.tunables.push(tunable);
        self.checksum = Once::new();
        self.stack_manifest = Once::new();
        #[cfg(std_io)]
        if let Some(cache) = &self.stack_plan_cache { cache.lock().clear(); }
        self
    }

    /// All candidate operations in this set, in registration order.
    pub fn autotunables(&self) -> impl Iterator<Item = &TuneFn<F, Output>> {
        self.tunables.iter().map(|tunable| &tunable.function)
    }

    /// Returns the [autotune plan](TunePlan) for the given set.
    pub(crate) fn plan(&self, key: &K) -> TunePlan {
        TunePlan::new(key, &self.tunables)
    }

    /// Returns the operation for the given index, matching the order returned by
    /// `autotunables`. Tunables are tried in order, so index 0 should be a good default.
    pub fn fastest(&self, fastest_index: usize) -> &TuneFn<F, Output> {
        &self.tunables[fastest_index].function
    }

    /// The identity of the candidate selected by the tuner.
    pub(crate) fn identity(&self, index: usize) -> &Id {
        self.tunables[index].identity()
    }

    /// Compute a checksum that invalidates outdated cached auto-tune results when the
    /// set of tunable names changes.
    pub fn compute_checksum(&self) -> String {
        // Preserve legacy cache identity when the new controller is not enabled.
        self.checksum.call_once(|| {
            let mut checksum = String::new();
            for tune in &self.tunables { checksum += &tune.function.name; }
            format!("{:x}", md5::compute(checksum))
        }).clone()
    }

    /// Separate, length-delimited manifest for the opt-in full-stack cache.
    pub fn stack_checksum(&self) -> String {
        String::from(self.stack_checksum_ref())
    }

    pub(crate) fn stack_checksum_ref(&self) -> &str {
        self.stack_manifest.call_once(|| {
            let mut checksum = format!("stack-v1:{}:{};reference={:?};", self.stack_revision.len(), self.stack_revision, self.stack_reference);
            for tune in &self.tunables {
                let _ = write!(checksum, "{}:{}", tune.function.name.len(), tune.function.name);
            }
            format!("{:x}", md5::compute(checksum))
        }).as_str()
    }

    /// Generate a key from a set of inputs
    pub fn generate_key<'a>(&self, inputs: &F::At<'a>) -> K {
        self.key_gen.generate(inputs)
    }

    /// Generate a set of test inputs from a key and reference inputs.
    pub fn generate_inputs<'a>(&self, key: &K, inputs: &F::At<'a>) -> F::At<'a> {
        self.input_gen.generate(key, inputs)
    }
}

#[cfg(serializable)]
/// Trait alias with support for persistent caching
pub trait AutotuneKey:
    Clone
    + Debug
    + PartialEq
    + Eq
    + Hash
    + Display
    + serde::Serialize
    + serde::de::DeserializeOwned
    + Send
    + Sync
    + 'static
{
}
#[cfg(not(serializable))]
/// Trait alias
pub trait AutotuneKey:
    Clone + Debug + PartialEq + Eq + Hash + Display + Send + Sync + 'static
{
}

impl AutotuneKey for String {}

#[cfg(all(test, std_io))]
mod static_plan_tests {
    use super::*;
    fn set() -> TunableSet<String, String, String> {
        TunableSet::new_cloning_inputs(|value: &String| value.clone())
            .with_static_stack_plan()
    }
    fn plan() -> Arc<StaticStackPlan> {
        Arc::new(StaticStackPlan { indices: alloc::vec![0],
            candidates: alloc::vec![super::super::stack::Candidate::new("reference")], key: "full-key".into() })
    }
    #[test]
    fn cache_requires_exact_generated_key_and_full_identity() {
        let set = set();
        let plan = plan();
        set.cache_stack_plan("shape".into(), "device/offset/stride".into(), plan.clone());
        assert!(Arc::ptr_eq(&set.cached_stack_plan(&"shape".into(), "device/offset/stride").unwrap(), &plan));
        assert!(set.cached_stack_plan(&"other-shape".into(), "device/offset/stride").is_none());
        assert!(set.cached_stack_plan(&"shape".into(), "other-device/offset/stride").is_none());
    }
    #[test]
    fn cache_is_bounded_and_duplicate_insert_does_not_evict() {
        let set = set();
        for index in 0..32 { set.cache_stack_plan(alloc::format!("{index}"), "binding".into(), plan()); }
        set.cache_stack_plan("0".into(), "binding".into(), plan());
        assert!(set.cached_stack_plan(&"0".into(), "binding").is_some());
        set.cache_stack_plan("32".into(), "binding".into(), plan());
        assert!(set.cached_stack_plan(&"0".into(), "binding").is_none());
        assert!(set.cached_stack_plan(&"1".into(), "binding").is_some());
    }
    #[test]
    fn changing_stack_contract_invalidates_plan_metadata() {
        let set = set();
        set.cache_stack_plan("shape".into(), "binding".into(), plan());
        let set = set.with_stack_tuning(0, "new-revision", |value: &String| value.clone());
        assert!(set.cached_stack_plan(&"shape".into(), "binding").is_none());
    }
}
