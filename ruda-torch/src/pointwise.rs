//! Storage-aware pointwise dispatch. No tensor-wide promotion allocations.
use super::{bf16, f16, client, finish_dispatch, kernels, CudaRuntime, View,
    LAUNCHES, DIRECT_POINTWISE_CALLS, Ordering};
use ruda_kernel::dsl::prelude::*;

fn vector_rows(view: &View) -> bool {
    !view.shape.is_empty() && view.len % 4 == 0
        && view.shape.last().unwrap() % 4 == 0 && view.strides.last() == Some(&1)
        && view.shape.iter().zip(&view.strides).take(view.shape.len() - 1)
            .all(|(&size, &stride)| size <= 1 || stride % 4 == 0)
        && view.handle.offset_start.unwrap_or(0) % (4 * super::element_bytes(view.dtype)) as u64 == 0
}

#[ruda]
fn vector_offset<F: Float + RudaElement>(input: &Tensor<Vector<F, Const<4>>>, position: usize,
    #[comptime] packed: bool) -> usize {
    let mut result = position;
    if !comptime!(packed) {
        let mut remaining = position * 4;
        result = 0;
        let mut axis = input.rank();
        while axis > 0 {
            axis -= 1;
            result += (remaining % input.shape(axis)) * input.stride(axis);
            remaining /= input.shape(axis);
        }
        result /= 4;
    }
    result
}

#[ruda(launch)]
fn vector_pointwise<A: Float + RudaElement, B: Float + RudaElement, O: Float + RudaElement>(
    a: &Tensor<Vector<A, Const<4>>>, b: &Tensor<Vector<B, Const<4>>>,
    out: &mut Array<Vector<O, Const<4>>>, scalar: f32,
    #[comptime] op: u32, #[comptime] packed_a: bool, #[comptime] packed_b: bool,
) {
    let position = ABSOLUTE_POS as usize;
    if position < out.len() {
        let x = Vector::<f32, Const<4>>::cast_from(a[vector_offset(a, position, packed_a)]);
        let mut y = Vector::<f32, Const<4>>::new(0.0);
        if comptime!(op == 1 || op == 2 || op == 8) {
            y = Vector::cast_from(b[vector_offset(b, position, packed_b)]);
        }
        let mut result = Vector::<f32, Const<4>>::empty();
        #[unroll]
        for lane in 0usize..4 {
            let value = x[lane];
            let mut output = value;
            if comptime!(op == 1) { output = value + scalar * y[lane]; }
            else if comptime!(op == 2) { output = value * y[lane]; }
            else if comptime!(op == 3) { if value < 0.0 { output = 0.0; } }
            else if comptime!(op == 8) { output = value / y[lane]; }
            else if comptime!(op == 9) { output = value.exp(); }
            else if comptime!(op == 10) { output = value.ln(); }
            else if comptime!(op == 11) { output = value.sqrt(); }
            else if comptime!(op == 12) { output = 1.0 / value.sqrt(); }
            else if comptime!(op == 13) { output = 1.0 / (1.0 + (-value).exp()); }
            else if comptime!(op == 14) { output = value / (1.0 + (-value).exp()); }
            else if comptime!(op == 17) { output = value.tanh(); }
            else if comptime!(op == 19) { output = value.sin(); }
            else if comptime!(op == 20) { output = value.cos(); }
            else if comptime!(op == 21) { output = value.abs(); }
            else if comptime!(op == 27) { output = value.recip(); }
            result[lane] = output;
        }
        out[position] = Vector::cast_from(result);
    }
}

pub(super) fn launch(op: u32, a: &View, b: &View, out: &View, scalar: f32) {
    if out.len == 0 { return; }
    let client = client();
    // ABSOLUTE_POS is a u32 in these compatibility kernels.
    assert!(out.len <= u32::MAX as usize, "native pointwise/scalar grid exceeds 32-bit indexing");
    let count = u32::try_from(out.len.div_ceil(128)).expect("launch grid overflow");
    let binary = matches!(op, 1 | 2 | 8);
    let vector = matches!(op, 1..=3 | 8..=14 | 17 | 19..=21 | 27)
        && vector_rows(a) && (!binary || vector_rows(b)) && out.contiguous()
        && out.len % 4 == 0
        && out.handle.offset_start.unwrap_or(0) % (4 * super::element_bytes(out.dtype)) as u64 == 0;
    macro_rules! run {
        ($a:ty, $b:ty, $o:ty) => {
            if vector {
                vector_pointwise::launch::<$a, $b, $o, CudaRuntime>(
                    &client, RudaCount::Static(out.len.div_ceil(512) as u32, 1, 1), RudaDim::new_1d(128),
                    a.arg(), b.arg(), out.array_arg(), scalar, op, a.contiguous(), b.contiguous())
            } else { kernels::pointwise::launch::<$a, $b, $o, CudaRuntime>(
                &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128),
                a.arg(), b.arg(), out.arg(), scalar, op) }
        };
    }
    unsafe {
        match (a.dtype, b.dtype, out.dtype) {
            (0, 0, 0) => run!(f32, f32, f32),
            (0, 0, 1) => run!(f32, f32, f16),
            (0, 0, 2) => run!(f32, f32, bf16),
            (0, 1, 0) => run!(f32, f16, f32),
            (0, 1, 1) => run!(f32, f16, f16),
            (0, 1, 2) => run!(f32, f16, bf16),
            (0, 2, 0) => run!(f32, bf16, f32),
            (0, 2, 1) => run!(f32, bf16, f16),
            (0, 2, 2) => run!(f32, bf16, bf16),
            (1, 0, 0) => run!(f16, f32, f32),
            (1, 0, 1) => run!(f16, f32, f16),
            (1, 0, 2) => run!(f16, f32, bf16),
            (1, 1, 0) => run!(f16, f16, f32),
            (1, 1, 1) => run!(f16, f16, f16),
            (1, 1, 2) => run!(f16, f16, bf16),
            (1, 2, 0) => run!(f16, bf16, f32),
            (1, 2, 1) => run!(f16, bf16, f16),
            (1, 2, 2) => run!(f16, bf16, bf16),
            (2, 0, 0) => run!(bf16, f32, f32),
            (2, 0, 1) => run!(bf16, f32, f16),
            (2, 0, 2) => run!(bf16, f32, bf16),
            (2, 1, 0) => run!(bf16, f16, f32),
            (2, 1, 1) => run!(bf16, f16, f16),
            (2, 1, 2) => run!(bf16, f16, bf16),
            (2, 2, 0) => run!(bf16, bf16, f32),
            (2, 2, 1) => run!(bf16, bf16, f16),
            (2, 2, 2) => run!(bf16, bf16, bf16),
            _ => panic!("unsupported RUDA pointwise dtype combination"),
        }
    }
    // Keep the existing synchronous bridge contract until streams and
    // allocation lifetime tracking are connected end-to-end.
    finish_dispatch(&client);
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    DIRECT_POINTWISE_CALLS.fetch_add(1, Ordering::Relaxed);
}
