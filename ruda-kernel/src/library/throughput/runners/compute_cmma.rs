use crate::dsl::prelude::*;
use ruda::runtime::throughput::{CmmaDims, ComputeCmmaConfig, KernelConfig, ThroughputKey};

use crate::library::throughput::LaunchConfig;

pub fn build_kernel<R: Runtime>(
    client: &ComputeClient<R>,
    key: ThroughputKey,
    cmma_config: ComputeCmmaConfig,
    config: LaunchConfig,
) -> KernelConfig {
    let client = client.fixed_execution_queue();
    let dtype = key.dtype();

    let ops_per_cmma = 2 * cmma_config.cmma_dims.num_elems();
    let out_bytes =
        cmma_config.cmma_dims.m * cmma_config.cmma_dims.n * cmma_config.accumulator_type.size();

    let sample = Box::new(move |iterations: usize| {
        let start = ruda_core::profile::Instant::now();
        let out = client.empty(out_bytes);
        unsafe {
            compute_cmma_throughput::launch_unchecked(
                &client,
                RudaCount::Static(config.ruda_count as u32, 1, 1),
                config.ruda_dim,
                config.vector_size,
                ArrayArg::from_raw_parts(out.clone(), cmma_config.cmma_dims.m * cmma_config.cmma_dims.n),
                iterations,
                cmma_config.cmma_dims,
                dtype.into(),
                cmma_config.accumulator_type.into(),
            )
        };
        super::memory_probe::synchronize(&client, &[&out])?;
        Ok(start.elapsed())
    });

    let planes_per_cube = config.ruda_dim.num_elems() as usize / config.plane_size;
    let ops_count = config.ruda_count * planes_per_cube * ops_per_cmma;

    KernelConfig {
        sample,
        ops_count,
        min_iterations: 1,
    }
}

#[ruda(launch_unchecked)]
pub fn compute_cmma_throughput<I: Numeric, ACC: Numeric, N: Size>(
    output: &mut Array<Vector<ACC, N>>,
    n_iter: usize,
    #[comptime] cmm_dims: CmmaDims,
    #[define(I)] _dtype: StorageType,
    #[define(ACC)] _acc: StorageType,
) {
    let CmmaDims { m, n, k } = cmm_dims;

    let a = cmma::Matrix::<I>::from_value(
        cmma::MatrixIdent::A,
        m,
        n,
        k,
        cmma::MatrixLayout::RowMajor,
        I::cast_from(1),
    );

    let b = cmma::Matrix::<I>::from_value(
        cmma::MatrixIdent::B,
        m,
        n,
        k,
        cmma::MatrixLayout::ColMajor,
        I::cast_from(1),
    );

    let acc = cmma::Matrix::<ACC>::from_value(
        cmma::MatrixIdent::Accumulator,
        m,
        n,
        k,
        cmma::MatrixLayout::Undefined,
        ACC::cast_from(0.0),
    );

    for _ in 0..n_iter {
        cmma::execute(&a, &b, &acc, &acc);
    }

    if ABSOLUTE_POS == 0 {
        cmma::store(&mut output.to_slice_mut(), &acc, n as u32, cmma::MatrixLayout::RowMajor);
    }
}
