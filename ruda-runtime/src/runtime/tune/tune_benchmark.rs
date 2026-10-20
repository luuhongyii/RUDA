use super::{AutotuneError, Evictor, TuneFn, TuneInputs};
use crate::runtime::{client::ComputeClient, backend::Runtime};
use alloc::string::ToString;
use alloc::vec::Vec;
use ruda_core::profile::ProfileDuration;

/// The trait to be implemented by an autotune output.
pub trait AutotuneOutput: Send + 'static {
    /// Fallible correctness gate for stack autotuning. `Ok(false)` explicitly means that this
    /// output has no validator; a successful launch alone is NOT a correctness check.
    /// Existing implementers need not add a method. Strict stack policy rejects unsupported output.
    fn validate_for_tuning(&self, _other: &Self, _absolute: f64, _relative: f64, _max_bytes: u64)
        -> Result<bool, alloc::string::String> { Ok(false) }

    #[cfg(feature = "runtime-autotune-checks")]
    /// Checks if the output of an autotune operation is the same as another one on the same
    /// problem.
    fn check_equivalence(&self, other: Self);
}

impl AutotuneOutput for () {
    #[cfg(feature = "runtime-autotune-checks")]
    fn check_equivalence(&self, _other: Self) {
        //
    }
}

/// Benchmark how long this operation takes for a number of samples.
///
/// Returns at least one duration, otherwise an error is returned.
pub fn tune_benchmark<'a, R: Runtime, F: TuneInputs, Out: AutotuneOutput>(
    operation: &TuneFn<F, Out>,
    inputs: <F as TuneInputs>::At<'a>,
    client: ComputeClient<R>,
) -> Result<Vec<ProfileDuration>, AutotuneError> {
    tune_benchmark_with_eviction(operation, inputs, client, None)
}

pub fn tune_benchmark_with_eviction<'a, R: Runtime, F: TuneInputs, Out: AutotuneOutput>(
    operation: &TuneFn<F, Out>,
    inputs: <F as TuneInputs>::At<'a>,
    client: ComputeClient<R>,
    evictor: Option<&mut Evictor<'_>>,
) -> Result<Vec<ProfileDuration>, AutotuneError> {
    // `scoped` holds exclusive device access for the whole benchmark loop and
    // accepts non-`'static` closures.
    client
        .clone()
        .exclusive(move || profile_exclusive(operation, inputs, client, evictor))
        .map_err(|err| AutotuneError::Unknown {
            name: operation.name.to_string(),
            err: err.to_string(),
        })?
}

impl<F: TuneInputs, Out: AutotuneOutput> TuneFn<F, Out> {
    /// Run the operation once without measuring it, to trigger compilation.
    ///
    /// Expects to already hold exclusive device access; the adaptive driver takes it once for
    /// the whole round robin rather than once per candidate.
    pub(crate) fn warmup_once<'a, R: Runtime>(
        &self,
        inputs: <F as TuneInputs>::At<'a>,
        client: &ComputeClient<R>,
    ) -> Result<(), AutotuneError> {
        // We make sure the server is in a correct state.
        let _errs = client.flush();

        // The profile is dropped without being resolved: a warmup only exists to surface a
        // failure to compile or launch, which is what the error carries — so nothing is
        // evicted for it, either.
        self.sample_once(inputs, client, None).map(|_| ())
    }

    /// Queue a single measured execution. See [`Self::warmup_once`] for the locking expectation.
    ///
    /// `evictor` runs first, outside the profiled region, so the sample reads what a real
    /// call reads rather than what the previous launch left in cache ([`Eviction`]). An
    /// eviction that fails is logged and the sample taken warm: the measurement is still
    /// worth more than none, and the failure is the eviction's, not the candidate's.
    ///
    /// [`Eviction`]: super::Eviction
    pub(crate) fn sample_once<'a, R: Runtime>(
        &self,
        inputs: <F as TuneInputs>::At<'a>,
        client: &ComputeClient<R>,
        evictor: Option<&mut Evictor<'_>>,
    ) -> Result<ProfileDuration, AutotuneError> {
        if let Some(evict) = evictor
            && let Err(err) = evict()
        {
            log::error!(
                "The eviction before a sample of `{}` failed, so the sample is measured on a warm cache.\n{err:?}",
                self.name
            );
        }
        // The output is returned so dead code elimination can't drop the work being profiled.
        let profiled = client.profile(move || self.execute(inputs), &self.name);

        match profiled {
            Ok((Ok(_), duration)) => Ok(duration),
            Ok((Err(err), _)) => Err(err),
            Err(err) => Err(AutotuneError::Unknown {
                name: self.name.to_string(),
                err: err.to_string(),
            }),
        }
    }
}

fn profile_exclusive<'a, R: Runtime, F: TuneInputs, Out: AutotuneOutput>(
    operation: &TuneFn<F, Out>,
    inputs: <F as TuneInputs>::At<'a>,
    client: ComputeClient<R>,
    mut evictor: Option<&mut Evictor<'_>>,
) -> Result<Vec<ProfileDuration>, AutotuneError> {
    let _real_run = crate::runtime::dry_run::RealRun::new();
    warmup(operation, inputs.clone(), client.clone())?;

    use crate::runtime::config::{RudaRuntimeConfig, RuntimeConfig};
    let (_, num_samples) = RudaRuntimeConfig::get().autotune.bench.samples();
    let mut durations = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        durations.push(operation.sample_once(inputs.clone(), &client, evictor.as_deref_mut())?);
    }

    if durations.is_empty() {
        Err(AutotuneError::InvalidSamples {
            name: operation.name.to_string(),
        })
    } else {
        Ok(durations)
    }
}

fn warmup<'a, R: Runtime, F: TuneInputs, Out: AutotuneOutput>(
    operation: &TuneFn<F, Out>,
    inputs: <F as TuneInputs>::At<'a>,
    client: ComputeClient<R>,
) -> Result<(), AutotuneError> {
    let num_warmup = 3;

    let mut num_errors = 0;
    let mut last_error = None;
    // We make sure the server is in a correct state.
    let _errs = client.flush();

    for _ in 0..num_warmup {
        let inputs = inputs.clone();
        let profiled = client.profile(move || operation.execute(inputs), &operation.name);

        match profiled {
            Ok((Ok(_), _)) => {}
            Ok((Err(err), _)) => return Err(err),
            Err(err) => {
                num_errors += 1;
                last_error = Some(err);
            }
        }
    }

    if num_errors < num_warmup {
        Ok(())
    } else {
        let msg = alloc::format!("{:?}", last_error.expect("all warmup profiles failed"));
        Err(AutotuneError::Unknown {
            name: operation.name.to_string(),
            err: msg,
        })
    }
}
