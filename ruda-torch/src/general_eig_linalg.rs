use super::*;
use rusolver::tensor::{complex::Matrix, general_eig, ldl};

pub(super) fn dispatch(vs: &[View], vectors: bool) {
    assert_eq!(vs.len(), 4);
    let (a, extra) = if matches!(vs[0].dtype, 0 | 9) {
        ldl::real_parts(&matrix(&vs[0])).expect("real general eigen components")
    } else {
        assert!(matches!(vs[0].dtype, 10 | 11));
        (Matrix { real: matrix(&scientific::component(&vs[0], false)), imag: matrix(&scientific::component(&vs[0], true)) }, 0)
    };
    let result = general_eig::eig(&a, vectors).expect("general complex eigen decomposition");
    scientific::copy_complex(&result.values.real, &result.values.imag, &vs[1]);
    if vectors { scientific::copy_complex(&result.vectors.real, &result.vectors.imag, &vs[2]); }
    copy_real(&result.info, &vs[3]);
    LAUNCHES.fetch_add((result.submitted_kernels + extra) as u64, Ordering::Relaxed);
}
