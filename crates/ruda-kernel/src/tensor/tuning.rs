use super::RudaTensor;
use crate::dsl::{Runtime, tune::AutotuneOutput};

struct Outputs<R: Runtime>(Vec<RudaTensor<R>>);

impl<R: Runtime> AutotuneOutput for Outputs<R> {
    fn validate_for_tuning(&self, other: &Self, absolute: f64, relative: f64, max_bytes: u64) -> Result<bool, String> {
        if self.0.len() != other.0.len() { return Err("native autotune output count differs".into()); }
        let bytes = self.0.iter().try_fold(0u64, |total, tensor| {
            let elements = tensor.meta.shape().iter().try_fold(1u64, |size, &extent| size.checked_mul(extent as u64))?;
            let bytes = if matches!(tensor.dtype, ruda_core::tensor::DType::QFloat(_)) { 4 } else { tensor.elem_size() as u64 };
            total.checked_add(elements.checked_mul(bytes)?.checked_mul(2)?)
        });
        if bytes.is_none_or(|bytes| bytes > max_bytes) { return Ok(false); }
        for (expected, actual) in self.0.iter().zip(&other.0) {
            if !expected.validate_for_tuning(actual, absolute, relative, max_bytes)? { return Ok(false); }
        }
        Ok(true)
    }

    #[cfg(feature = "device-tensor-autotune-checks")]
    fn check_equivalence(&self, other: Self) {
        assert_eq!(self.0.len(), other.0.len());
        for (expected, actual) in self.0.iter().zip(other.0) { expected.check_equivalence(actual); }
    }
}

/// Execute an actual native candidate through the installed shared controller.
/// Trial inputs are immutable original operands; each candidate must allocate its own outputs and scratch.
/// Every visible output is checked, including saved statistics and requested training derivatives.
/// Without a shared controller, returns `None` without launching or changing the original route.
pub fn execute_variants<R, C, F>(inputs: Vec<RudaTensor<R>>, operation: &'static str, semantics: String,
    candidates: Vec<(&'static str, C)>, execute: F) -> Result<Option<Vec<RudaTensor<R>>>, String>
where R: Runtime, C: Copy + Send + Sync + 'static,
    F: Fn(&[RudaTensor<R>], C) -> Result<Vec<RudaTensor<R>>, String> + Send + Sync + 'static {
    #[cfg(all(any(feature = "std", feature = "frontend-std"), any(target_os = "windows", target_os = "linux", target_os = "macos", target_os = "android")))]
    {
        use crate::dsl::tune::{Tunable, TunableSet, stack::{stack_autotuner, try_execute_stack}};
        use std::sync::Arc;
        if stack_autotuner().is_none() { return Ok(None); }
        let input = inputs.first().ok_or("native autotune requires actual input operands")?;
        let client = input.client.clone();
        let device = format!("{:?}", client.device_id());
        let signature = format!("{semantics};operands={:?}", inputs.iter().map(RudaTensor::autotune_signature).collect::<Vec<_>>());
        let key = signature.clone();
        let mut set = TunableSet::<String, Vec<RudaTensor<R>>, Outputs<R>>::new_cloning_inputs(move |_: &Vec<RudaTensor<R>>| key.clone())
            .with_stack_tuning(0, operation, move |_: &Vec<RudaTensor<R>>| signature.clone());
        let execute = Arc::new(execute);
        for (name, config) in candidates {
            let execute = execute.clone();
            set = set.with(Tunable::new(name, move |values: Vec<RudaTensor<R>>| execute(&values, config).map(Outputs)));
        }
        let output = try_execute_stack(operation, &device, &client, Arc::new(set), inputs).map_err(|error| format!("{error:?}"))?;
        Ok(Some(output.0))
    }
    #[cfg(not(all(any(feature = "std", feature = "frontend-std"), any(target_os = "windows", target_os = "linux", target_os = "macos", target_os = "android"))))]
    {
        let _ = (inputs, operation, semantics, candidates, execute);
        Ok(None)
    }
}

pub fn execute_variants_prepared<R, C, F>(inputs: Vec<RudaTensor<R>>, operation: &'static str, semantics: String,
    candidates: Vec<(&'static str, C)>, execute: F) -> Result<Option<Vec<RudaTensor<R>>>, String>
where R: Runtime, C: Copy + core::fmt::Debug + Eq + core::hash::Hash + Send + Sync + 'static,
    F: Fn(&[RudaTensor<R>], C) -> Result<Vec<RudaTensor<R>>, String> + Send + Sync + 'static {
    #[cfg(all(any(feature = "std", feature = "frontend-std"), any(target_os = "windows", target_os = "linux", target_os = "macos", target_os = "android")))]
    {
        use crate::dsl::tune::{Tunable, TunableSet, stack::{stack_autotuner, fields,
            prepare_stack_operation, try_execute_prepared_stack, PreparedStackOperation}};
        use std::{any::{Any, TypeId}, collections::{BTreeMap, hash_map::DefaultHasher},
            hash::{Hash, Hasher}, sync::{Arc, OnceLock, RwLock}};
        type CacheKey = (TypeId, TypeId, TypeId, u16, u16, u64, u64, String);
        type Cache = BTreeMap<CacheKey, Arc<dyn Any + Send + Sync>>;
        type Live<R: Runtime, F> = (Vec<RudaTensor<R>>, Arc<F>);
        type Prepared<R: Runtime, C, F: Send + Sync + 'static> = (Vec<(&'static str, C)>, PreparedStackOperation<String, Live<R,F>, Outputs<R>>);
        static CACHE: OnceLock<RwLock<Cache>> = OnceLock::new();
        let Some(tuner) = stack_autotuner() else { return Ok(None); };
        let input = inputs.first().ok_or("native autotune requires actual input operands")?;
        let client = input.client.clone();
        let id = client.device_id();
        let signature = format!("{semantics};operands={:?}", inputs.iter().map(RudaTensor::autotune_signature).collect::<Vec<_>>());
        let mut hasher = DefaultHasher::new(); candidates.hash(&mut hasher);
        let identity = (TypeId::of::<R>(), TypeId::of::<C>(), TypeId::of::<F>(), id.type_id, id.index_id,
            client.properties_fingerprint(), hasher.finish(), fields(&[operation, &signature]));
        let cache = CACHE.get_or_init(|| RwLock::new(BTreeMap::new()));
        let retained = cache.read().unwrap_or_else(|p| p.into_inner()).get(&identity).cloned()
            .and_then(|value| value.downcast::<Prepared<R,C,F>>().ok())
            .filter(|value| value.0 == candidates);
        let live = (inputs, Arc::new(execute));
        let prepared = match retained {
            Some(prepared) => prepared,
            None => {
                let signature = format!("{signature};configurations={candidates:?}");
                let key = signature.clone();
                let mut set = TunableSet::<String, Live<R,F>, Outputs<R>>::new_cloning_inputs(move |_: &Live<R,F>| key.clone())
                    .with_stack_tuning(0, operation, move |_: &Live<R,F>| signature.clone());
                for &(name, config) in &candidates {
                    set = set.with(Tunable::new(name, move |(values, execute): Live<R,F>| execute(&values, config).map(Outputs)));
                }
                let prepared = prepare_stack_operation(operation, &format!("{:?}", client.device_id()),
                    &client, Arc::new(set), &live).map_err(|error| format!("{error:?}"))?;
                let prepared = Arc::new((candidates, prepared));
                let mut entries = cache.write().unwrap_or_else(|p| p.into_inner());
                if entries.len() >= tuner.policy().capacity {
                    if let Some(oldest) = entries.keys().next().cloned() { entries.remove(&oldest); }
                }
                entries.insert(identity, prepared.clone());
                prepared
            }
        };
        let output = try_execute_prepared_stack(operation, &client, &prepared.1, live)
            .map_err(|error| format!("{error:?}"))?;
        Ok(Some(output.0))
    }
    #[cfg(not(all(any(feature = "std", feature = "frontend-std"), any(target_os = "windows", target_os = "linux", target_os = "macos", target_os = "android"))))]
    {
        let _ = (inputs, operation, semantics, candidates, execute);
        Ok(None)
    }
}

/// Whether this target has an installed shared controller. Avoid preparing expensive keys when disabled.
pub fn is_enabled() -> bool {
    #[cfg(all(any(feature = "std", feature = "frontend-std"), any(target_os = "windows", target_os = "linux", target_os = "macos", target_os = "android")))]
    { crate::dsl::tune::stack::stack_autotuner().is_some() }
    #[cfg(not(all(any(feature = "std", feature = "frontend-std"), any(target_os = "windows", target_os = "linux", target_os = "macos", target_os = "android"))))]
    { false }
}

/// Original launch plus actual eligible one-dimensional block-size alternatives.
pub fn elementwise_candidates<R: Runtime>(input: &RudaTensor<R>) -> Vec<(&'static str, u32)> {
    let hardware = &input.client.properties().hardware;
    let maximum = hardware.max_ruda_dim.0.min(hardware.max_units_per_ruda);
    let mut candidates = vec![("original_launch", 0)];
    for (name, units) in [("units_64", 64), ("units_128", 128), ("units_256", 256), ("units_512", 512)] {
        if units <= maximum { candidates.push((name, units)); }
    }
    candidates
}
