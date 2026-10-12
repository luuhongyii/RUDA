use super::scientific::{copy_real, real_tensor};
use super::*;
use ruda_core::tensor::DType;
use ruda_kernel::tensor::{
    RudaTensor, allocation::empty_device_contiguous_dtype, contiguous::into_contiguous,
    permutation::swap_dims, readback::into_data, reshape::reshape,
};
use rusolver::tensor as solver;
type Tensor = RudaTensor<CudaRuntime>;
#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_linalg_api_version() -> u32 {
    1
}
fn matrix(v: &View) -> Tensor {
    assert!(
        v.dtype == 0 || v.dtype == 9,
        "linalg supports real float32/float64 only"
    );
    assert!(v.shape.len() >= 2, "matrix dimensions required");
    let r = v.shape.len();
    let batch = v.shape[..r - 2].iter().product::<usize>();
    let m = v.shape[r - 2];
    let n = v.shape[r - 1];
    reshape(into_contiguous(real_tensor(v)), [batch, m, n].into())
}
fn pivots(v: &View, batch: usize, k: usize, add: i32) -> Tensor {
    assert_eq!(v.dtype, 5, "pivots must be int32");
    assert_eq!(v.len, batch * k);
    let input = into_contiguous(real_tensor(v));
    let out = empty_device_contiguous_dtype(client(), cuda_device(), [batch, k].into(), DType::I32);
    if v.len > 0 {
        unsafe {
            pivot_shift::launch::<CudaRuntime>(
                &client(),
                RudaCount::Static(v.len.div_ceil(128) as u32, 1, 1),
                RudaDim::new_1d(128),
                input.into_array_arg(),
                out.clone().into_array_arg(),
                add,
            );
        }
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
    }
    out
}
fn emit_pivots(p: &Tensor, out: &View) {
    assert_eq!(out.dtype, 5);
    let t =
        empty_device_contiguous_dtype(client(), cuda_device(), p.meta.shape().clone(), DType::I32);
    let n = p.meta.num_elements();
    if n > 0 {
        unsafe {
            pivot_shift::launch::<CudaRuntime>(
                &client(),
                RudaCount::Static(n.div_ceil(128) as u32, 1, 1),
                RudaDim::new_1d(128),
                p.clone().into_array_arg(),
                t.clone().into_array_arg(),
                1,
            );
        }
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
    }
    copy_real(&t, out)
}
fn empty_info(batch: usize) -> Tensor {
    let info = empty_device_contiguous_dtype(client(), cuda_device(), [batch].into(), DType::I32);
    if batch > 0 {
        unsafe {
            zero_info::launch::<CudaRuntime>(
                &client(),
                RudaCount::Static(batch.div_ceil(128) as u32, 1, 1),
                RudaDim::new_1d(128),
                info.clone().into_array_arg(),
            );
        }
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
    }
    info
}
fn check_info(v: &View) {
    assert_eq!(v.dtype, 5);
    if v.len == 0 {
        return;
    }
    let d = block_on(into_data(into_contiguous(real_tensor(v)))).expect("linalg status readback");
    DOWNLOAD.fetch_add(v.len as u64 * 4, Ordering::Relaxed);
    for (batch, &code) in d
        .to_vec::<i32>()
        .expect("int32 linalg info")
        .iter()
        .enumerate()
    {
        assert!(code == 0, "linalg batch {batch} failed with info={code}");
    }
}
fn zero_options() -> solver::LargeSolverOptions {
    solver::LargeSolverOptions {
        absolute_tolerance: 0.0,
        relative_tolerance: 0.0,
        ..Default::default()
    }
}
fn jacobi_options(dtype: DType, scalars: &[f64], ints: &[i64]) -> solver::DeviceJacobiOptions {
    let tolerance =
        scalars
            .first()
            .copied()
            .unwrap_or(if dtype == DType::F32 { 1e-6 } else { 1e-12 });
    solver::DeviceJacobiOptions {
        absolute_tolerance: 0.0,
        relative_tolerance: tolerance,
        symmetry_tolerance: 0.0,
        max_sweeps: ints
            .get(2)
            .copied()
            .unwrap_or(64)
            .try_into()
            .expect("Jacobi sweep budget"),
    }
}
/// Commands and descriptor order are shared with the Python linalg adapter.
/// 0 LU factor; 1 LU reuse solve; 2 Cholesky factor; 3 triangular solve;
/// 4 Cholesky reuse solve; 5 QR; 6 QR least squares; 7 eigh; 8 SVD;
/// 9 inverse; 10 determinant; 11 apply stored Q; 12 geqrf; 13 Cholesky inverse;
/// 15 unpack LU factors and inverse permutation;
/// 255 synchronizes/reads only the small integer info tensor.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_linalg(
    op: u32,
    ds: *const Descriptor,
    count: usize,
    ints: *const i64,
    ni: usize,
    scalars: *const f64,
    ns: usize,
) -> i32 {
    checked(|| {
        assert!(count > 0 && !ds.is_null());
        let ds = unsafe { std::slice::from_raw_parts(ds, count) };
        let vs: Vec<View> = ds.iter().map(|d| unsafe { View::read(d) }).collect();
        let p = if ni == 0 {
            &[][..]
        } else {
            assert!(!ints.is_null());
            unsafe { std::slice::from_raw_parts(ints, ni) }
        };
        let f = if ns == 0 {
            &[][..]
        } else {
            assert!(!scalars.is_null());
            unsafe { std::slice::from_raw_parts(scalars, ns) }
        };
        let flag = |i: usize, default: bool| p.get(i).map_or(default, |&x| x != 0);
        if op == 255 {
            assert_eq!(count, 1);
            check_info(&vs[0]);
            return;
        }
        let a = matrix(&vs[0]);
        let (batch, m, n) = (a.meta.shape()[0], a.meta.shape()[1], a.meta.shape()[2]);
        match op {
            0 => {
                assert_eq!(count, 4);
                let r = solver::lu_factor_blocked(&a, flag(0, true), zero_options())
                    .expect("LU factor");
                copy_real(&r.packed_lu, &vs[1]);
                emit_pivots(&r.pivots, &vs[2]);
                copy_real(&r.info, &vs[3]);
                LAUNCHES.fetch_add(r.submitted_kernels as u64, Ordering::Relaxed);
            }
            1 => {
                assert_eq!(count, 5);
                let pv = pivots(&vs[1], batch, n, -1);
                let b = matrix(&vs[2]);
                let r = solver::lu_solve_factor(&a, &pv, &b, flag(0, false))
                    .expect("LU factor reuse solve");
                copy_real(&r.solution, &vs[3]);
                copy_real(&r.info, &vs[4]);
                LAUNCHES.fetch_add(r.submitted_kernels as u64, Ordering::Relaxed);
            }
            2 => {
                assert_eq!(count, 3);
                let r =
                    solver::cholesky_factor_blocked(&a, flag(0, false)).expect("Cholesky factor");
                copy_real(&r.factor, &vs[1]);
                copy_real(&r.info, &vs[2]);
                LAUNCHES.fetch_add(r.submitted_kernels as u64, Ordering::Relaxed);
            }
            3 => {
                assert_eq!(count, 4);
                let b = matrix(&vs[1]);
                let r =
                    solver::triangular_solve(&a, &b, flag(0, true), flag(1, false), flag(2, false))
                        .expect("triangular solve");
                copy_real(&r.solution, &vs[2]);
                copy_real(&r.info, &vs[3]);
                LAUNCHES.fetch_add(r.submitted_kernels as u64, Ordering::Relaxed);
            }
            4 => {
                assert_eq!(count, 4);
                let b = matrix(&vs[1]);
                let r = solver::cholesky_solve_factor(&a, &b, flag(0, false))
                    .expect("Cholesky factor reuse solve");
                copy_real(&r.solution, &vs[2]);
                copy_real(&r.info, &vs[3]);
                LAUNCHES.fetch_add(r.submitted_kernels as u64, Ordering::Relaxed);
            }
            5 => {
                assert_eq!(count, 6);
                let mode = p.first().copied().unwrap_or(0);
                assert!(mode >= 0 && mode <= 2);
                let r = solver::qr_factor_blocked(&a, zero_options()).expect("Householder QR");
                let qcols = if mode == 1 { m } else { m.min(n) };
                if mode != 2 {
                    copy_real(
                        &solver::qr_form_q(&r.packed, &r.tau, qcols).expect("form Q"),
                        &vs[1],
                    );
                }
                copy_real(
                    &solver::qr_extract_r(&r.packed, if mode == 1 { m } else { m.min(n) })
                        .expect("extract R"),
                    &vs[2],
                );
                copy_real(&r.packed, &vs[3]);
                copy_real(&r.tau, &vs[4]);
                copy_real(&r.info, &vs[5]);
                LAUNCHES.fetch_add(r.submitted_kernels as u64 + 2, Ordering::Relaxed);
            }
            6 => {
                assert_eq!(count, 5);
                let b = matrix(&vs[1]);
                let r = solver::least_squares_qr(&a, &b).expect("Householder gels least squares");
                copy_real(&r.solution, &vs[2]);
                copy_real(&r.residuals, &vs[3]);
                copy_real(&r.info, &vs[4]);
                LAUNCHES.fetch_add(r.submitted_kernels as u64, Ordering::Relaxed);
            }
            7 => {
                assert_eq!(count, 4);
                assert_eq!(m, n);
                if n == 0 || batch == 0 {
                    copy_real(&empty_info(batch), &vs[3]);
                } else {
                    let sym =
                        solver::symmetric_from_triangle(&a, flag(0, false)).expect("eigh UPLO");
                    let r = solver::symmetric_eigen_device(&sym, jacobi_options(a.dtype, f, p))
                        .expect("eigh");
                    copy_real(&r.values, &vs[1]);
                    if flag(1, true) {
                        copy_real(&r.vectors, &vs[2]);
                    }
                    copy_real(&r.info, &vs[3]);
                    LAUNCHES.fetch_add(r.submitted_kernels as u64, Ordering::Relaxed);
                }
            }
            8 => {
                assert_eq!(count, 5);
                let full = flag(0, false);
                let uv = flag(1, true);
                if batch == 0 || m == 0 || n == 0 {
                    if uv && full {
                        copy_real(&solver::identity_matrix(&a, batch, m), &vs[1]);
                        copy_real(&solver::identity_matrix(&a, batch, n), &vs[3]);
                    }
                    copy_real(&empty_info(batch), &vs[4]);
                } else {
                    let r = solver::svd_device(&a, jacobi_options(a.dtype, f, p))
                        .expect("one-sided Jacobi SVD");
                    copy_real(&r.singular_values, &vs[2]);
                    if uv {
                        let u = if full && m > m.min(n) {
                            solver::orthogonal_complete(&r.u).expect("full U")
                        } else {
                            r.u
                        };
                        let v = if full && n > m.min(n) {
                            solver::orthogonal_complete(&r.v).expect("full V")
                        } else {
                            r.v
                        };
                        copy_real(&u, &vs[1]);
                        copy_real(&swap_dims(v, 1, 2), &vs[3]);
                    }
                    copy_real(&r.info, &vs[4]);
                    LAUNCHES.fetch_add(r.submitted_kernels as u64, Ordering::Relaxed);
                }
            }
            9 => {
                assert_eq!(count, 5);
                let f =
                    solver::lu_factor_blocked(&a, true, zero_options()).expect("inverse LU factor");
                let r = solver::lu_inverse_factor(&f).expect("inverse solve");
                copy_real(&r.solution, &vs[1]);
                copy_real(&f.packed_lu, &vs[2]);
                emit_pivots(&f.pivots, &vs[3]);
                copy_real(&f.info, &vs[4]);
                LAUNCHES.fetch_add(
                    (f.submitted_kernels + r.submitted_kernels) as u64,
                    Ordering::Relaxed,
                );
            }
            10 => {
                assert_eq!(count, 6);
                let f = solver::lu_factor_blocked(&a, true, zero_options())
                    .expect("determinant LU factor");
                let r = solver::determinant_from_lu(&f).expect("determinant");
                copy_real(&f.packed_lu, &vs[1]);
                emit_pivots(&f.pivots, &vs[2]);
                copy_real(&r.determinant, &vs[3]);
                copy_real(&r.sign, &vs[4]);
                copy_real(&r.log_absolute, &vs[5]);
                LAUNCHES.fetch_add(f.submitted_kernels as u64 + 1, Ordering::Relaxed);
            }
            11 => {
                assert_eq!(count, 4);
                let tau = into_contiguous(real_tensor(&vs[1]));
                let b = matrix(&vs[2]);
                let r =
                    solver::qr_apply_q(&a, &tau, &b, flag(0, false)).expect("apply Householder Q");
                copy_real(&r, &vs[3]);
                LAUNCHES.fetch_add(1, Ordering::Relaxed);
            }
            12 => {
                assert_eq!(count, 4);
                let r = solver::qr_factor_blocked(&a, zero_options()).expect("geqrf");
                copy_real(&r.packed, &vs[1]);
                copy_real(&r.tau, &vs[2]);
                copy_real(&r.info, &vs[3]);
                LAUNCHES.fetch_add(r.submitted_kernels as u64, Ordering::Relaxed);
            }
            13 => {
                assert_eq!(count, 3);
                let r =
                    solver::cholesky_inverse_factor(&a, flag(0, false)).expect("Cholesky inverse");
                copy_real(&r.solution, &vs[1]);
                copy_real(&r.info, &vs[2]);
                LAUNCHES.fetch_add(r.submitted_kernels as u64, Ordering::Relaxed);
            }
            15 => {
                assert_eq!(count, 5);
                let pv = pivots(&vs[1], batch, m.min(n), -1);
                let r = solver::lu_unpack_factor(&a, &pv, flag(0, true), flag(1, true))
                    .expect("LU unpack");
                copy_real(&r.permutation, &vs[2]);
                copy_real(&r.lower, &vs[3]);
                copy_real(&r.upper, &vs[4]);
                LAUNCHES.fetch_add(1, Ordering::Relaxed);
            }
            _ => panic!("unsupported linalg command"),
        }
        finish_dispatch(&client());
    })
}
#[ruda(launch)]
fn pivot_shift(input: &Array<i32>, out: &mut Array<i32>, add: i32) {
    let pos = ABSOLUTE_POS;
    if pos < out.len() {
        out[pos] = input[pos] + add;
    }
}
#[ruda(launch)]
fn zero_info(out: &mut Array<i32>) {
    let pos = ABSOLUTE_POS;
    if pos < out.len() {
        out[pos] = 0;
    }
}
