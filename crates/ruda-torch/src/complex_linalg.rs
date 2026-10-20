use super::*;
use rusolver::tensor::complex as complex_solver;
use rusolver::tensor::complex_qr;
use rusolver::tensor::complex_spectral;

fn input(v: &View) -> complex_solver::Matrix<CudaRuntime> {
    assert!(matches!(v.dtype, 10 | 11), "complex64/complex128 matrix required");
    complex_solver::Matrix {
        real: matrix(&scientific::component(v, false)),
        imag: matrix(&scientific::component(v, true)),
    }
}

fn emit(value: &complex_solver::Matrix<CudaRuntime>, out: &View) {
    scientific::copy_complex(&value.real, &value.imag, out);
}

fn reflectors(v: &View, batch: usize) -> complex_solver::Matrix<CudaRuntime> {
    let k = *v.shape.last().expect("tau dimension required"); assert_eq!(v.len, batch * k);
    complex_solver::Matrix {
        real: reshape(into_contiguous(real_tensor(&scientific::component(v, false))), [batch, 1, k].into()),
        imag: reshape(into_contiguous(real_tensor(&scientific::component(v, true))), [batch, 1, k].into()),
    }
}

pub(super) fn dispatch(op: u32, vs: &[View], flags: &[i64], scalars: &[f64]) {
    let a = input(&vs[0]);
    let shape = a.real.meta.shape(); let (batch, m, n) = (shape[0], shape[1], shape[2]);
    let flag = |index: usize, default: bool| flags.get(index).map_or(default, |&value| value != 0);
    let launches = match op {
        0 => {
            assert_eq!(vs.len(), 4);
            let factor = complex_solver::lu_factor(&a, flag(0, true)).expect("complex LU factor");
            emit(&factor.packed, &vs[1]); emit_pivots(&factor.pivots, &vs[2]); copy_real(&factor.info, &vs[3]);
            factor.submitted_kernels
        }
        1 => {
            assert_eq!(vs.len(), 5); assert_eq!(m, n);
            let p = pivots(&vs[1], batch, n, -1); let b = input(&vs[2]);
            let result = complex_solver::lu_solve(&a, &p, &b, flag(0, false)).expect("complex LU solve");
            emit(&result.solution, &vs[3]); copy_real(&result.info, &vs[4]); result.submitted_kernels
        }
        2 => {
            assert_eq!(vs.len(), 3);
            let result = complex_solver::cholesky_factor(&a, flag(0, false)).expect("complex Cholesky factor");
            emit(&result.factor, &vs[1]); copy_real(&result.info, &vs[2]); result.submitted_kernels
        }
        3 | 4 => {
            assert_eq!(vs.len(), 4); let b = input(&vs[1]);
            let result = if op == 3 {
                complex_solver::triangular_solve(&a, &b, flag(0, true), flag(1, false), false, flag(2, false))
            } else { complex_solver::cholesky_solve(&a, &b, flag(0, false)) }.expect("complex factor solve");
            emit(&result.solution, &vs[2]); copy_real(&result.info, &vs[3]); result.submitted_kernels
        }
        5 => {
            assert_eq!(vs.len(), 6);
            let mode = flags.first().copied().unwrap_or(0); assert!((0..=2).contains(&mode));
            let factor = complex_qr::factor(&a).expect("complex Householder QR");
            let rows = if mode == 1 { m } else { m.min(n) };
            let mut submitted = factor.submitted_kernels;
            if mode != 2 {
                let q = complex_qr::form_q(&factor.packed, &factor.tau, rows).expect("form complex Q");
                emit(&q.value, &vs[1]); submitted += q.submitted_kernels;
            }
            let r = complex_qr::extract_r(&factor.packed, rows).expect("extract complex R");
            emit(&r.value, &vs[2]); emit(&factor.packed, &vs[3]); emit(&factor.tau, &vs[4]); copy_real(&factor.info, &vs[5]);
            submitted + r.submitted_kernels
        }
        6 => {
            assert_eq!(vs.len(), 5); let b = input(&vs[1]);
            let result = complex_qr::least_squares(&a, &b).expect("complex Householder gels least squares");
            emit(&result.solution, &vs[2]); copy_real(&result.residuals, &vs[3]); copy_real(&result.info, &vs[4]);
            result.submitted_kernels
        }
        7 => {
            assert_eq!(vs.len(), 4);
            let result = complex_spectral::eigh(&a, flag(0, false), flag(1, true), jacobi_options(a.real.dtype, scalars, flags))
                .expect("complex Hermitian Jacobi eigen decomposition");
            copy_real(&result.values, &vs[1]);
            if flag(1, true) { emit(&result.vectors, &vs[2]); }
            copy_real(&result.info, &vs[3]); result.submitted_kernels
        }
        8 => {
            assert_eq!(vs.len(), 5);
            let result = complex_spectral::svd(&a, flag(0, false), flag(1, true), jacobi_options(a.real.dtype, scalars, flags))
                .expect("complex one-sided Jacobi SVD");
            copy_real(&result.singular_values, &vs[2]);
            if flag(1, true) { emit(&result.u, &vs[1]); emit(&result.vh, &vs[3]); }
            copy_real(&result.info, &vs[4]); result.submitted_kernels
        }
        9 => {
            assert_eq!(vs.len(), 5); assert_eq!(m, n);
            let factor = complex_solver::lu_factor(&a, true).expect("complex inverse LU");
            let identity = complex_solver::identity(&a).expect("complex identity");
            let result = complex_solver::lu_solve(&factor.packed, &factor.pivots, &identity, false).expect("complex inverse solve");
            emit(&result.solution, &vs[1]); emit(&factor.packed, &vs[2]); emit_pivots(&factor.pivots, &vs[3]); copy_real(&factor.info, &vs[4]);
            factor.submitted_kernels + result.submitted_kernels + usize::from(batch * n != 0)
        }
        10 => {
            assert_eq!(vs.len(), 6); assert_eq!(m, n);
            let factor = complex_solver::lu_factor(&a, true).expect("complex determinant LU");
            let result = complex_solver::determinant(&factor).expect("complex determinant");
            emit(&factor.packed, &vs[1]); emit_pivots(&factor.pivots, &vs[2]);
            emit(&result.determinant, &vs[3]); emit(&result.sign, &vs[4]); copy_real(&result.log_absolute, &vs[5]);
            factor.submitted_kernels + result.submitted_kernels
        }
        11 => {
            assert_eq!(vs.len(), 4); assert_eq!(vs[0].dtype, vs[1].dtype);
            let tau = reflectors(&vs[1], batch);
            let b = input(&vs[2]);
            let result = complex_qr::apply_q(&a, &tau, &b, flag(0, false)).expect("apply complex Householder Q");
            emit(&result.value, &vs[3]); result.submitted_kernels
        }
        12 => {
            assert_eq!(vs.len(), 4);
            let result = complex_qr::factor(&a).expect("complex geqrf");
            emit(&result.packed, &vs[1]); emit(&result.tau, &vs[2]); copy_real(&result.info, &vs[3]);
            result.submitted_kernels
        }
        13 => {
            assert_eq!(vs.len(), 3); assert_eq!(m, n);
            let identity = complex_solver::identity(&a).expect("complex identity");
            let result = complex_solver::cholesky_solve(&a, &identity, flag(0, false)).expect("complex Cholesky inverse");
            emit(&result.solution, &vs[1]); copy_real(&result.info, &vs[2]);
            result.submitted_kernels + usize::from(batch * n != 0)
        }
        14 => {
            assert_eq!(vs.len(), 3); assert_eq!(vs[0].dtype, vs[1].dtype);
            let tau = reflectors(&vs[1], batch);
            let result = complex_qr::form_q(&a, &tau, n).expect("complex Householder product");
            emit(&result.value, &vs[2]); result.submitted_kernels
        }
        15 => {
            assert_eq!(vs.len(), 5); let p = pivots(&vs[1], batch, m.min(n), -1);
            let result = complex_solver::lu_unpack(&a, &p, flag(0, true), flag(1, true)).expect("complex LU unpack");
            emit(&result.permutation, &vs[2]); emit(&result.lower, &vs[3]); emit(&result.upper, &vs[4]); result.submitted_kernels
        }
        _ => panic!("complex linalg command is not implemented"),
    };
    LAUNCHES.fetch_add(launches as u64, Ordering::Relaxed);
}
