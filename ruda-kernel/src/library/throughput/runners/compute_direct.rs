use crate::dsl::prelude::*;
use ruda_core::ir::ElemType;
use crate::dsl::frontend::fma;
use ruda::runtime::throughput::KernelConfig;

use crate::library::throughput::LaunchConfig;

pub fn build_kernel<R: Runtime>(client: &ComputeClient<R>, dtype: ElemType, config: LaunchConfig) -> KernelConfig {
    let client = client.fixed_execution_queue();

    let use_fma = matches!(dtype, ElemType::Float(_));

    let sample = Box::new(move |iterations: usize| {
        let start = ruda_core::profile::Instant::now();
        let out = client.empty(config.vector_size * dtype.size());
        unsafe {
            compute_direct_throughput::launch_unchecked(
                &client,
                RudaCount::Static(config.ruda_count as u32, 1, 1),
                config.ruda_dim,
                config.vector_size,
                ArrayArg::from_raw_parts(out.clone(), config.vector_size),
                iterations,
                use_fma,
                dtype.into(),
            )
        };
        super::memory_probe::synchronize(&client, &[&out])?;
        Ok(start.elapsed())
    });

    // `CHAINS` independent accumulators per lane, each retiring one fma (two flops) or one mul.
    let ops_per_chain = if use_fma { 2 } else { 1 };
    let ops_count = ops_per_chain
        * CHAINS
        * config.ruda_count
        * config.ruda_dim.num_elems() as usize
        * config.vector_size;

    KernelConfig {
        sample,
        ops_count,
        min_iterations: 1,
    }
}

/// Independent accumulator chains per lane to hide arithmetic latency.
const CHAINS: usize = 4;

/// Vectors the kernel keeps live across its loop: the chains and their two operands.
pub(crate) const LIVE_VECTORS: usize = CHAINS + 2;

#[ruda(launch_unchecked)]
pub fn compute_direct_throughput<I: Numeric, N: Size>(
    output: &mut Array<Vector<I, N>>,
    n_iter: usize,
    #[comptime] use_fma: bool,
    #[define(I)] _dtype: StorageType,
) {
    let tid = I::cast_from(ABSOLUTE_POS);

    let mut b = Vector::<I, N>::empty();
    let mut c = Vector::<I, N>::empty();

    let mut s0 = Vector::<I, N>::empty();
    let mut s1 = Vector::<I, N>::empty();
    let mut s2 = Vector::<I, N>::empty();
    let mut s3 = Vector::<I, N>::empty();

    // Give every lane and chain a distinct seed to prevent folding.
    let lanes = b.vector_size();
    #[unroll]
    for lane in 0..lanes {
        let offset = I::cast_from(lane);
        b[lane] = tid + offset + I::cast_from(1);
        c[lane] = tid + offset;

        s0[lane] = offset + I::cast_from(1);
        s1[lane] = offset + I::cast_from(2);
        s2[lane] = offset + I::cast_from(3);
        s3[lane] = offset + I::cast_from(4);
    }

    for _ in 0..n_iter {
        s0 = step(s0, b, c, use_fma);
        s1 = step(s1, b, c, use_fma);
        s2 = step(s2, b, c, use_fma);
        s3 = step(s3, b, c, use_fma);
    }

    let sum = s0 + s1 + s2 + s3;

    if ABSOLUTE_POS == 0 {
        output[0] = sum;
    }
}

/// Retires one arithmetic op per chain: an fma (two flops) for floats, otherwise a mul
/// (the slowest integer op, giving a lower bound).
#[ruda]
fn step<I: Numeric, N: Size>(
    s: Vector<I, N>,
    b: Vector<I, N>,
    c: Vector<I, N>,
    #[comptime] use_fma: bool,
) -> Vector<I, N> {
    if use_fma { fma(s, b, c) } else { s * b }
}
