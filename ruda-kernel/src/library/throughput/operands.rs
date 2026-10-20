use ruda_core::ir::{ElemType, FloatKind, features::Features};
use ruda::runtime::{backend::Runtime, client::ComputeClient, throughput::ComputeCmmaConfig};

use crate::library::throughput::compute_direct;

/// The operand types and vector widths an arithmetic ceiling is measured in.
pub(super) struct Arithmetic;

impl Arithmetic {
    /// Without an fma of its own for `dtype` a device emulates one per
    /// operation, and a kernel converts and accumulates in f32 rather than pay
    /// that, so both are measured and the faster kept.
    pub(super) fn dtypes<R: Runtime>(client: &ComputeClient<R>, dtype: ElemType) -> alloc::vec::Vec<ElemType> {
        let mut dtypes = alloc::vec![dtype];

        if let Some(accumulator) = Self::promoted(dtype)
            && client.properties().features.supports_type(accumulator)
        {
            dtypes.push(accumulator);
        }

        dtypes
    }

    /// The probe issues no loads, so it sweeps down from the widest vector kept in registers.
    pub(super) fn widths<R: Runtime>(client: &ComputeClient<R>, dtype: ElemType) -> alloc::vec::Vec<usize> {
        client
            .properties()
            .vector_sizes_in_registers(dtype.size(), compute_direct::LIVE_VECTORS)
            .collect()
    }

    /// What a kernel with `dtype` operands accumulates in when the device has no
    /// arithmetic of its own for them, or `None` where `dtype` is already the
    /// widest of the two.
    fn promoted(dtype: ElemType) -> Option<ElemType> {
        let accumulator = ElemType::Float(FloatKind::F32);
        let narrower_float =
            matches!(dtype, ElemType::Float(_)) && dtype.size() < accumulator.size();

        narrower_float.then_some(accumulator)
    }
}

/// The cooperative matrix a compute-cmma probe launches.
pub(super) struct CooperativeMatrix;

impl CooperativeMatrix {
    /// A non-empty capability list says the device has tensor hardware, not this
    /// shape of it. `mma` is not consulted: the probe issues `cmma::execute`.
    pub(super) fn implemented(
        features: &Features,
        dtype: ElemType,
        config: ComputeCmmaConfig,
    ) -> bool {
        features.matmul.cmma.iter().any(|it| {
            it.a_type == dtype.into()
                && it.b_type == dtype.into()
                && it.cd_type == config.accumulator_type.into()
                && it.m as usize == config.cmma_dims.m
                && it.n as usize == config.cmma_dims.n
                && it.k as usize == config.cmma_dims.k
        })
    }
}
