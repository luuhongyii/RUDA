use crate::dsl::prelude::*;
use ruda::runtime::{server::ProfileError, throughput::ThroughputError};

pub fn build_kernel<R: Runtime>(
    client: &ComputeClient<R>,
    _key: ruda::runtime::throughput::ThroughputKey,
    _config: super::super::LaunchConfig,
) -> ruda::runtime::throughput::KernelConfig {
    let client = client.fixed_execution_queue();
    let sample = alloc::boxed::Box::new(move |iterations: usize| {
        let input = client.empty(core::mem::size_of::<i32>());
        let output = client.empty(core::mem::size_of::<i32>());

        // Opened before the window and read only if the device times nothing:
        // the fallback has to span the same launches, so it starts with them.
        let start = ruda_core::profile::Instant::now();
        let mut launched = false;

        let profiled = client.profile(
            || unsafe {
                launched = true;
                for _ in 0..iterations {
                    launch_overhead::launch_unchecked(
                        &client,
                        RudaCount::new_single(),
                        RudaDim::new_single(),
                        1,
                        ArrayArg::from_raw_parts(input.clone(), 1),
                        ArrayArg::from_raw_parts(output.clone(), 1),
                        ruda_core::ir::ElemType::Int(ruda_core::ir::IntKind::I32).into(),
                    );
                }
            },
            "launch_overhead",
        );

        // What the device made of the window, when it made anything of it. Two
        // ways it makes nothing, both seen on Metal and neither a broken probe:
        //
        // * [`ProfileError::NotMeasured`] — the window resolved no timestamped
        //   pass at all, so there is no span to report.
        // * `None` — the pair came back unwritten or out of order, which
        //   `stop_profile` declines to pass off as a duration.
        //
        // Any other profile error is a window that did not run as asked, and it
        // is not this probe's to interpret either.
        let measured = match profiled {
            Ok((_, duration)) => {
                ruda_core::future::block_on(duration.into_future()).map(|ticks| ticks.duration())
            }
            Err(ProfileError::NotMeasured { .. }) if launched => None,
            Err(_) => return Err(ThroughputError::Launch),
        };

        // The host clock over the same launches, which is what every other
        // probe in this module times with. A launch costs the caller what it
        // adds to the wall clock, and `iterations` of them amortize the single
        // sync this pays for, so the host span is a measurement of the same
        // thing rather than a stand-in for one.
        //
        super::memory_probe::synchronize(&client, &[&output])?;
        Ok(measured.unwrap_or_else(|| start.elapsed()))
    });

    ruda::runtime::throughput::KernelConfig {
        sample,
        ops_count: 1,
        min_iterations: 1,
    }
}

#[ruda(launch_unchecked)]
pub fn launch_overhead<I: Numeric, N: Size>(
    input: &Array<Vector<I, N>>,
    output: &mut Array<Vector<I, N>>,
    #[define(I)] _dtype: StorageType,
) {
    if ABSOLUTE_POS == 0 {
        output[0] = input[0];
    }
}
