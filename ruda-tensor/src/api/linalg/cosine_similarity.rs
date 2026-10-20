use crate::api::ElementConversion;
use crate::api::FloatDType;
use crate::api::backend::Backend;
use crate::api::Tensor;

use super::l2_norm;

/// Computes the cosine similarity between two tensors along a specified dimension.
///
/// Calculates the cosine of the angle between inputs as their dot product divided
/// by the product of their L2 norms.
///
/// # Arguments
///
/// * `x1` - First input tensor
/// * `x2` - Second input tensor
/// * `dim` - Dimension along which to compute the similarity
///   (negative indices allowed: -1 for last dimension)
/// * `eps` - Small value to avoid division by zero (default: dtype's smallest positive normal)
///
/// # Returns
///
/// Tensor containing the cosine similarity between x1 and x2
pub fn cosine_similarity<B: Backend, const D: usize>(
    x1: Tensor<B, D>,
    x2: Tensor<B, D>,
    dim: i32,
    eps: Option<B::FloatElem>,
) -> Tensor<B, D> {
    let eps = eps.map(|value| value.elem::<f64>()).unwrap_or_else(|| {
        x1
            .dtype()
            .finfo()
            .unwrap_or(FloatDType::F32.finfo())
            .min_positive
    });

    // Convert negative dimension to positive
    let dim_idx = if dim < 0 { D as i32 + dim } else { dim };
    assert!((0..D as i32).contains(&dim_idx), "Cosine Similarity dimension is out of bounds");
    let dim_idx = dim_idx as usize;

    // Compute L2 norms: ||x1|| and ||x2||
    let norm_x1 = l2_norm(x1.clone(), dim_idx).clamp_min(eps);
    let norm_x2 = l2_norm(x2.clone(), dim_idx).clamp_min(eps);

    // Return the cosine similarity (dot product divided by the product of norms)
    ((x1 / norm_x1) * (x2 / norm_x2)).sum_dim(dim_idx)
}
