use super::*;
use rusolver::tensor::{complex::Matrix, ldl, matrix_exp};

fn input_matrix(v: &View) -> Tensor {
    if !matches!(v.dtype, 1 | 2) { return matrix(v); }
    assert!(v.shape.len() >= 2, "matrix dimensions required");
    let r = v.shape.len();
    let batch = v.shape[..r - 2].iter().product::<usize>();
    reshape(into_contiguous(real_tensor(v)), [batch, v.shape[r - 2], v.shape[r - 1]].into())
}

pub(super) fn dispatch(vs: &[View]) {
    assert_eq!(vs.len(), 2); assert_eq!(vs[0].dtype, vs[1].dtype);
    let complex_input = matches!(vs[0].dtype, 10 | 11 | 14);
    let (a, extra) = if complex_input {
        (Matrix { real: input_matrix(&scientific::component(&vs[0], false)), imag: input_matrix(&scientific::component(&vs[0], true)) }, 0)
    } else if matches!(vs[0].dtype, 1 | 2) {
        let real = input_matrix(&vs[0]);
        (Matrix { imag: real.clone(), real }, 0)
    } else { ldl::real_parts(&matrix(&vs[0])).expect("real matrix exponential components") };
    let result = matrix_exp::exp(&a, complex_input).expect("matrix exponential");
    if complex_input { scientific::copy_complex(&result.value.real, &result.value.imag, &vs[1]); }
    else { copy_real(&result.value.real, &vs[1]); }
    LAUNCHES.fetch_add((result.submitted_kernels + extra) as u64, Ordering::Relaxed);
}
