use super::*;
use rusolver::tensor::{complex::Matrix, ldl as ldl_solver};

fn input(v: &View) -> (Matrix<CudaRuntime>, usize) {
    if matches!(v.dtype, 0 | 9) { return ldl_solver::real_parts(&matrix(v)).expect("real LDL components"); }
    assert!(matches!(v.dtype, 10 | 11));
    (Matrix { real: matrix(&scientific::component(v, false)), imag: matrix(&scientific::component(v, true)) }, 0)
}

fn emit(value: &Matrix<CudaRuntime>, out: &View) {
    if matches!(out.dtype, 0 | 9) { copy_real(&value.real, out); }
    else { scientific::copy_complex(&value.real, &value.imag, out); }
}

pub(super) fn dispatch(op: u32, vs: &[View], flags: &[i64]) {
    let (a, mut submitted) = input(&vs[0]); let hermitian = flags.first().copied().unwrap_or(0) != 0;
    if op == 16 {
        assert_eq!(vs.len(), 4);
        let result = ldl_solver::factor(&a, hermitian).expect("Bunch-Kaufman LDL factor");
        emit(&result.packed, &vs[1]); copy_real(&result.pivots, &vs[2]); copy_real(&result.info, &vs[3]); submitted += result.submitted_kernels;
    } else {
        assert_eq!(op, 17); assert_eq!(vs.len(), 5); assert_eq!(vs[1].dtype, 5);
        let (b, launches) = input(&vs[2]); submitted += launches;
        let p = into_contiguous(real_tensor(&vs[1]));
        let result = ldl_solver::solve(&a, &p, &b, hermitian).expect("packed LDL solve");
        emit(&result.solution, &vs[3]); copy_real(&result.info, &vs[4]); submitted += result.submitted_kernels;
    }
    LAUNCHES.fetch_add(submitted as u64, Ordering::Relaxed);
}
